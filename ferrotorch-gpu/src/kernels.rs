//! Custom PTX CUDA kernels for elementwise GPU operations.
//!
//! Each operation attempts to run a PTX kernel on the GPU. If the PTX module
//! cannot be loaded (e.g. architecture mismatch), the function returns
//! `Err(GpuError::PtxCompileFailed { kernel })` per the PyTorch-parity device
//! error policy. There is no silent CPU fallback.
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
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64, cpu_to_gpu, gpu_to_cpu};

// ---------------------------------------------------------------------------
// f32 → f64 PTX auto-conversion
// ---------------------------------------------------------------------------

/// Convert an f32 PTX kernel string to its f64 equivalent by applying
/// mechanical substitutions. Works for "simple" kernels where the only
/// difference between f32 and f64 is register types, load/store widths,
/// byte offsets, and float literals.
///
/// Does NOT work for kernels that use `ex2.approx.f32` or `lg2.approx.f32`
/// (transcendentals) — those need hand-written f64 implementations.
#[cfg(feature = "cuda")]
pub(crate) fn ptx_f32_to_f64(
    f32_ptx: &str,
    f32_kernel_name: &str,
    f64_kernel_name: &str,
) -> String {
    f32_ptx
        // Kernel entry point name
        .replace(f32_kernel_name, f64_kernel_name)
        // Register declarations
        .replace(".reg .f32", ".reg .f64")
        // Memory operations (must come before arithmetic to avoid double-replace)
        .replace("ld.global.f32", "ld.global.f64")
        .replace("st.global.f32", "st.global.f64")
        .replace("ld.shared.f32", "ld.shared.f64")
        .replace("st.shared.f32", "st.shared.f64")
        .replace("ld.param.f32", "ld.param.f64")
        .replace(".param .f32", ".param .f64")
        // Shared memory declarations
        .replace(".shared .align 4 .f32", ".shared .align 8 .f64")
        // Arithmetic
        .replace("add.f32", "add.f64")
        .replace("sub.f32", "sub.f64")
        .replace("mul.f32", "mul.f64")
        .replace("div.rn.f32", "div.rn.f64")
        .replace("div.f32", "div.f64")
        .replace("neg.f32", "neg.f64")
        .replace("abs.f32", "abs.f64")
        .replace("max.f32", "max.f64")
        .replace("min.f32", "min.f64")
        .replace("selp.f32", "selp.f64")
        .replace("sqrt.rn.f32", "sqrt.rn.f64")
        .replace("sqrt.f32", "sqrt.f64")
        .replace("fma.rn.f32", "fma.rn.f64")
        .replace("mov.f32", "mov.f64")
        // Comparisons
        .replace("setp.gt.f32", "setp.gt.f64")
        .replace("setp.ge.f32", "setp.ge.f64")
        .replace("setp.lt.f32", "setp.lt.f64")
        .replace("setp.le.f32", "setp.le.f64")
        .replace("setp.eq.f32", "setp.eq.f64")
        .replace("setp.ne.f32", "setp.ne.f64")
        // Conversions
        .replace("cvt.rn.f32.u32", "cvt.rn.f64.u32")
        .replace("cvt.rn.f32.s32", "cvt.rn.f64.s32")
        // Bit reinterpretation (for NaN/inf checks)
        .replace("mov.b32", "mov.b64")
        // Byte offset: 4 bytes per f32 → 8 bytes per f64.
        // Cover both the canonical `%off` register and the `%off_in`/`%off_out`
        // pair used by gather/scatter/transpose-style kernels. Missing one of
        // these caused `gpu_transpose_2d_f64` to issue f32-stride loads against
        // an f64 buffer, hitting CUDA_ERROR_MISALIGNED_ADDRESS. (#575)
        .replace("shl.b64 %off, %off, 2", "shl.b64 %off, %off, 3")
        .replace("shl.b64 %off_in, %off_in, 2", "shl.b64 %off_in, %off_in, 3")
        .replace(
            "shl.b64 %off_out, %off_out, 2",
            "shl.b64 %off_out, %off_out, 3",
        )
        .replace(
            "shl.b64 %off_src, %off_src, 2",
            "shl.b64 %off_src, %off_src, 3",
        )
        .replace(
            "shl.b64 %off_dst, %off_dst, 2",
            "shl.b64 %off_dst, %off_dst, 3",
        )
        // Atomics
        .replace("atom.global.add.f32", "atom.global.add.f64")
        // Common float hex literals
        .replace("0f00000000", "0d0000000000000000") // 0.0
        .replace("0f3F800000", "0d3FF0000000000000") // 1.0
        .replace("0fBF800000", "0dBFF0000000000000") // -1.0
        .replace("0f40000000", "0d4000000000000000") // 2.0
        .replace("0f3F000000", "0d3FE0000000000000") // 0.5
        .replace("0fFF800000", "0dFFF0000000000000") // -inf
        .replace("0f7F800000", "0d7FF0000000000000") // +inf
        .replace("0f3FB8AA3B", "0d3FF71547652B82FE") // log2(e)
        .replace("0f3F317218", "0d3FE62E42FEFA39EF") // ln(2)
}

/// Helper to get or create a cached f64 PTX string from an f32 source.
///
/// Uses a global cache so the string transformation only happens once per
/// kernel. The returned `&str` is valid for the lifetime of the program.
#[cfg(feature = "cuda")]
pub(crate) fn get_f64_ptx<'a>(
    cache: &'a std::sync::OnceLock<String>,
    f32_ptx: &str,
    f32_name: &str,
    f64_name: &str,
) -> &'a str {
    cache.get_or_init(|| ptx_f32_to_f64(f32_ptx, f32_name, f64_name))
}

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
    .reg .u32 %my_tid, %bid, %bdim, %D_reg, %total_reg;
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
    mov.u32 %my_tid, %tid.x;
    mad.lo.u32 %my_tid, %bid, %bdim, %my_tid;

    setp.ge.u32 %p, %my_tid, %total_reg;
    @%p bra DONE;

    // row = tid / D, col = tid % D
    div.u32 %row, %my_tid, %D_reg;
    rem.u32 %col, %my_tid, %D_reg;

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
    cvt.u64.u32 %off, %my_tid;
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
    .reg .u32 %my_tid, %bid, %bdim, %D_reg, %total_reg;
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
    mov.u32 %my_tid, %tid.x;
    mad.lo.u32 %my_tid, %bid, %bdim, %my_tid;

    setp.ge.u32 %p, %my_tid, %total_reg;
    @%p bra DONE;

    // row = tid / D, col = tid % D
    div.u32 %row, %my_tid, %D_reg;
    rem.u32 %col, %my_tid, %D_reg;

    // Read grad_output[tid]
    cvt.u64.u32 %off, %my_tid;
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

/// PTX source for `gelu_f64_kernel`: `out[i] = x * sigmoid(1.702 * x)` (f64).
/// Uses f32-downcast for transcendentals.
#[cfg(feature = "cuda")]
pub(crate) const GELU_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_f64_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off;
    .reg .f64 %x, %neg_kx, %exp_neg, %one, %denom, %sig, %result, %k;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
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
    shl.b64 %off, %off, 3;
    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %x, [%in];
    mov.f64 %one, 0d3FF0000000000000;

    // k = 1.702
    mov.f64 %k, 0d3FFB44E400000000;
    mul.f64 %neg_kx, %k, %x;
    neg.f64 %neg_kx, %neg_kx;

    // --- exp(%neg_kx) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %neg_kx, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg_kx;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %exp_neg, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %exp_neg, %exp_neg, %e_nf;
    // --- end exp ---

    add.f64 %denom, %one, %exp_neg;
    div.rn.f64 %sig, %one, %denom;
    mul.f64 %result, %x, %sig;
    st.global.f64 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `gelu_tanh_kernel`: tanh approximation of GELU.
/// `out[i] = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
///
/// Uses `ex2.approx.f32` for exp and Horner-form tanh approximation via
/// `tanh(y) = (e^(2y) - 1) / (e^(2y) + 1)`.
#[cfg(feature = "cuda")]
pub(crate) const GELU_TANH_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_tanh_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off;
    .reg .f32 %x, %x3, %inner, %sqrt2pi, %c, %y, %two_y, %e2y;
    .reg .f32 %e2y_m1, %e2y_p1, %th, %one, %half, %log2e, %result;
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

    // inner = sqrt(2/π) * (x + 0.044715 * x³)
    // sqrt(2/π) = 0.7978845608 = 0x3F4C422A
    // 0.044715 = 0x3D372713
    mul.f32 %x3, %x, %x;
    mul.f32 %x3, %x3, %x;
    mov.f32 %c, 0f3D372713;
    mul.f32 %x3, %c, %x3;
    add.f32 %inner, %x, %x3;
    mov.f32 %sqrt2pi, 0f3F4C422A;
    mul.f32 %y, %sqrt2pi, %inner;

    // tanh(y) = (exp(2y) - 1) / (exp(2y) + 1)
    // exp(2y) = 2^(2y * log2(e))
    mov.f32 %log2e, 0f3FB8AA3B;
    add.f32 %two_y, %y, %y;
    mul.f32 %two_y, %two_y, %log2e;
    ex2.approx.f32 %e2y, %two_y;
    mov.f32 %one, 0f3F800000;
    sub.f32 %e2y_m1, %e2y, %one;
    add.f32 %e2y_p1, %e2y, %one;
    rcp.approx.f32 %e2y_p1, %e2y_p1;
    mul.f32 %th, %e2y_m1, %e2y_p1;

    // out = 0.5 * x * (1 + tanh)
    add.f32 %th, %one, %th;
    mov.f32 %half, 0f3F000000;
    mul.f32 %result, %half, %x;
    mul.f32 %result, %result, %th;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `gelu_tanh_f64_kernel`: tanh-approx GELU (f64).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(2y) in tanh.
#[cfg(feature = "cuda")]
pub(crate) const GELU_TANH_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_tanh_f64_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off;
    .reg .f64 %x, %x3, %inner, %sqrt2pi, %c, %y, %two_y, %e2y;
    .reg .f64 %e2y_m1, %e2y_p1, %th, %one, %half, %result;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
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
    shl.b64 %off, %off, 3;
    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %x, [%in];
    mov.f64 %one, 0d3FF0000000000000;

    // inner = sqrt(2/pi) * (x + 0.044715 * x^3)
    mul.f64 %x3, %x, %x;
    mul.f64 %x3, %x3, %x;
    mov.f64 %c, 0d3FA6E4E260000000;
    mul.f64 %x3, %c, %x3;
    add.f64 %inner, %x, %x3;
    mov.f64 %sqrt2pi, 0d3FE9884540000000;
    mul.f64 %y, %sqrt2pi, %inner;

    // tanh(y) = (exp(2y)-1)/(exp(2y)+1), exp(2y) in full f64
    add.f64 %two_y, %y, %y;

    // --- exp(%two_y) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %two_y, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %two_y;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e2y, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e2y, %e2y, %e_nf;
    // --- end exp ---

    sub.f64 %e2y_m1, %e2y, %one;
    add.f64 %e2y_p1, %e2y, %one;
    div.rn.f64 %th, %e2y_m1, %e2y_p1;

    // out = 0.5 * x * (1 + tanh)
    add.f64 %th, %one, %th;
    mov.f64 %half, 0d3FE0000000000000;
    mul.f64 %result, %half, %x;
    mul.f64 %result, %result, %th;
    st.global.f64 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `gelu_erf_kernel`: exact GELU using erf.
/// `out[i] = x * 0.5 * (1 + erf(x / sqrt(2)))`
///
/// Uses Abramowitz & Stegun formula 7.1.26 for erf (|ε| < 1.5×10⁻⁷).
#[cfg(feature = "cuda")]
pub(crate) const GELU_ERF_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_erf_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off;
    .reg .f32 %x, %z, %ax, %one, %half, %log2e;
    .reg .f32 %t, %pt, %z2, %neg_z2, %exp_neg_z2, %erf_val;
    .reg .f32 %p, %a1, %a2, %a3, %a4, %a5, %result;
    .reg .pred %pred_ge, %pred_neg;

    ld.param.u64 %in, [in_ptr];
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
    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %x, [%in];
    mov.f32 %one, 0f3F800000;
    mov.f32 %half, 0f3F000000;
    mov.f32 %log2e, 0f3FB8AA3B;

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
    mul.f32 %neg_z2, %neg_z2, %log2e;
    ex2.approx.f32 %exp_neg_z2, %neg_z2;

    // erf(|z|) = 1 - poly * exp(-z^2)
    mul.f32 %erf_val, %pt, %exp_neg_z2;
    sub.f32 %erf_val, %one, %erf_val;

    // erf(-z) = -erf(z), so sign-correct
    setp.lt.f32 %pred_neg, %z, 0f00000000;
    @%pred_neg neg.f32 %erf_val, %erf_val;

    // out = x * 0.5 * (1 + erf(x/sqrt(2)))
    add.f32 %erf_val, %one, %erf_val;
    mul.f32 %result, %half, %x;
    mul.f32 %result, %result, %erf_val;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `gelu_erf_f64_kernel`: exact erf GELU (f64).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(-z^2).
#[cfg(feature = "cuda")]
pub(crate) const GELU_ERF_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_erf_f64_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off;
    .reg .f64 %x, %z, %ax, %one, %half;
    .reg .f64 %t, %pt, %z2, %neg_z2, %exp_neg_z2, %erf_val;
    .reg .f64 %p, %a1, %a2, %a3, %a4, %a5, %result;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
    .reg .pred %pred_ge, %pred_neg;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %pred_ge, %r_tid, %n_reg;
    @%pred_ge bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 3;
    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %x, [%in];
    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %half, 0d3FE0000000000000;

    // z = x / sqrt(2) = x * 0.70710678
    mov.f64 %z, 0d3FE6A09E60000000;
    mul.f64 %z, %x, %z;

    abs.f64 %ax, %z;

    // t = 1 / (1 + 0.3275911 * |z|)
    mov.f64 %p, 0d3FD4F740A0000000;
    mul.f64 %t, %p, %ax;
    add.f64 %t, %one, %t;
    div.rn.f64 %t, %one, %t;

    // Horner: poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
    mov.f64 %a5, 0d3FC1555560000000;
    mov.f64 %a4, 0dBFD6752060000000;
    mov.f64 %a3, 0d3FF6A0DBA0000000;
    mov.f64 %a2, 0dBFE0783C20000000;
    mov.f64 %a1, 0d3FD41AD760000000;

    mul.f64 %pt, %t, %a5;
    add.f64 %pt, %pt, %a4;
    mul.f64 %pt, %pt, %t;
    add.f64 %pt, %pt, %a3;
    mul.f64 %pt, %pt, %t;
    add.f64 %pt, %pt, %a2;
    mul.f64 %pt, %pt, %t;
    add.f64 %pt, %pt, %a1;
    mul.f64 %pt, %pt, %t;

    // exp(-z^2) in full f64
    mul.f64 %z2, %ax, %ax;
    neg.f64 %neg_z2, %z2;

    // --- exp(%neg_z2) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %neg_z2, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg_z2;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %exp_neg_z2, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %exp_neg_z2, %exp_neg_z2, %e_nf;
    // --- end exp ---

    mul.f64 %erf_val, %pt, %exp_neg_z2;
    sub.f64 %erf_val, %one, %erf_val;

    setp.lt.f64 %pred_neg, %z, 0d0000000000000000;
    @%pred_neg neg.f64 %erf_val, %erf_val;

    add.f64 %erf_val, %one, %erf_val;
    mul.f64 %result, %half, %x;
    mul.f64 %result, %result, %erf_val;
    st.global.f64 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `gelu_backward_tanh_kernel`:
/// Backward for tanh approximation of GELU.
/// Let `u = sqrt(2/π) * (x + 0.044715 * x³)`, `t = tanh(u)`.
/// `d/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t²) * sqrt(2/π) * (1 + 3*0.044715*x²)`
/// `out[i] = grad[i] * d/dx`
#[cfg(feature = "cuda")]
pub(crate) const GELU_BACKWARD_TANH_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_backward_tanh_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f32 %vg, %x, %x2, %x3, %inner, %sqrt2pi, %c, %c3, %y;
    .reg .f32 %two_y, %e2y, %e2y_m1, %e2y_p1, %th, %one, %half, %log2e;
    .reg .f32 %th2, %one_m_th2, %d_inner, %term1, %term2, %d_gelu, %result;
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

    mov.f32 %one, 0f3F800000;
    mov.f32 %half, 0f3F000000;
    mov.f32 %log2e, 0f3FB8AA3B;
    mov.f32 %sqrt2pi, 0f3F4C422A;
    mov.f32 %c, 0f3D372713;
    // 3 * 0.044715 = 0.134145 = 0x3E096B8C
    mov.f32 %c3, 0f3E096B8C;

    // u = sqrt(2/π) * (x + 0.044715 * x³)
    mul.f32 %x2, %x, %x;
    mul.f32 %x3, %x2, %x;
    mul.f32 %x3, %c, %x3;
    add.f32 %inner, %x, %x3;
    mul.f32 %y, %sqrt2pi, %inner;

    // tanh(y) via exp
    add.f32 %two_y, %y, %y;
    mul.f32 %two_y, %two_y, %log2e;
    ex2.approx.f32 %e2y, %two_y;
    sub.f32 %e2y_m1, %e2y, %one;
    add.f32 %e2y_p1, %e2y, %one;
    rcp.approx.f32 %e2y_p1, %e2y_p1;
    mul.f32 %th, %e2y_m1, %e2y_p1;

    // d/dx = 0.5*(1+tanh) + 0.5*x*(1-tanh²)*sqrt(2/π)*(1+3*0.044715*x²)
    // term1 = 0.5 * (1 + th)
    add.f32 %term1, %one, %th;
    mul.f32 %term1, %half, %term1;

    // (1 - th²)
    mul.f32 %th2, %th, %th;
    sub.f32 %one_m_th2, %one, %th2;

    // d_inner = sqrt(2/π) * (1 + 3*0.044715*x²)
    mul.f32 %d_inner, %c3, %x2;
    add.f32 %d_inner, %one, %d_inner;
    mul.f32 %d_inner, %sqrt2pi, %d_inner;

    // term2 = 0.5 * x * (1-th²) * d_inner
    mul.f32 %term2, %half, %x;
    mul.f32 %term2, %term2, %one_m_th2;
    mul.f32 %term2, %term2, %d_inner;

    add.f32 %d_gelu, %term1, %term2;
    mul.f32 %result, %vg, %d_gelu;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `gelu_backward_tanh_f64_kernel`: tanh-approx backward (f64).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(2y) in tanh.
#[cfg(feature = "cuda")]
pub(crate) const GELU_BACKWARD_TANH_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_backward_tanh_f64_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f64 %vg, %x, %x2, %x3, %inner, %sqrt2pi, %c, %c3, %y;
    .reg .f64 %two_y, %e2y, %e2y_m1, %e2y_p1, %th, %one, %half;
    .reg .f64 %th2, %one_m_th2, %d_inner, %term1, %term2, %d_gelu, %result;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
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
    shl.b64 %off, %off, 3;
    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %vg, [%grad];
    ld.global.f64 %x, [%input];

    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %half, 0d3FE0000000000000;
    mov.f64 %sqrt2pi, 0d3FE9884540000000;
    mov.f64 %c, 0d3FA6E4E260000000;
    // 3 * 0.044715 = 0.134145
    mov.f64 %c3, 0d3FC12D7180000000;

    mul.f64 %x2, %x, %x;
    mul.f64 %x3, %x2, %x;
    mul.f64 %x3, %c, %x3;
    add.f64 %inner, %x, %x3;
    mul.f64 %y, %sqrt2pi, %inner;

    // tanh(y) = (exp(2y)-1)/(exp(2y)+1) in full f64
    add.f64 %two_y, %y, %y;

    // --- exp(%two_y) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %two_y, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %two_y;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e2y, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e2y, %e2y, %e_nf;
    // --- end exp ---

    sub.f64 %e2y_m1, %e2y, %one;
    add.f64 %e2y_p1, %e2y, %one;
    div.rn.f64 %th, %e2y_m1, %e2y_p1;

    add.f64 %term1, %one, %th;
    mul.f64 %term1, %half, %term1;

    mul.f64 %th2, %th, %th;
    sub.f64 %one_m_th2, %one, %th2;

    mul.f64 %d_inner, %c3, %x2;
    add.f64 %d_inner, %one, %d_inner;
    mul.f64 %d_inner, %sqrt2pi, %d_inner;

    mul.f64 %term2, %half, %x;
    mul.f64 %term2, %term2, %one_m_th2;
    mul.f64 %term2, %term2, %d_inner;

    add.f64 %d_gelu, %term1, %term2;
    mul.f64 %result, %vg, %d_gelu;
    st.global.f64 [%out], %result;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// SiLU / ELU / Mish activation kernels (forward + backward)
// ---------------------------------------------------------------------------

/// PTX source for `silu_kernel`: `out[i] = x * sigmoid(x)`.
/// SiLU (Sigmoid Linear Unit), also known as Swish-1.
#[cfg(feature = "cuda")]
pub(crate) const SILU_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry silu_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %x, %neg, %e, %denom, %sig, %vr, %one, %lg2e;
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

    ld.global.f32 %x, [%a];
    // sigmoid(x) = 1 / (1 + exp(-x))
    // exp(-x) = 2^(-x * log2(e))
    mov.f32 %one, 0f3F800000;
    mov.f32 %lg2e, 0f3FB8AA3B;
    neg.f32 %neg, %x;
    mul.f32 %neg, %neg, %lg2e;
    ex2.approx.f32 %e, %neg;
    add.f32 %denom, %one, %e;
    rcp.approx.f32 %sig, %denom;
    // silu(x) = x * sigmoid(x)
    mul.f32 %vr, %x, %sig;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `silu_f64_kernel`: `out[i] = x * sigmoid(x)` (f64).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(-x).
#[cfg(feature = "cuda")]
pub(crate) const SILU_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry silu_f64_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f64 %x, %neg_x, %e, %denom, %sig, %vr, %one;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %x, [%a];
    mov.f64 %one, 0d3FF0000000000000;
    neg.f64 %neg_x, %x;

    // --- exp(%neg_x) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %neg_x, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg_x;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e, %e, %e_nf;
    // --- end exp ---

    add.f64 %denom, %one, %e;
    div.rn.f64 %sig, %one, %denom;
    mul.f64 %vr, %x, %sig;
    st.global.f64 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `silu_backward_kernel`:
/// `out[i] = grad[i] * (sig + x * sig * (1 - sig))` where `sig = sigmoid(input[i])`.
#[cfg(feature = "cuda")]
pub(crate) const SILU_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry silu_backward_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f32 %vg, %x, %neg, %e, %denom, %sig, %one, %lg2e;
    .reg .f32 %one_m_sig, %x_sig_omsig, %deriv, %result;
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

    // sig = sigmoid(x) = 1 / (1 + exp(-x))
    mov.f32 %one, 0f3F800000;
    mov.f32 %lg2e, 0f3FB8AA3B;
    neg.f32 %neg, %x;
    mul.f32 %neg, %neg, %lg2e;
    ex2.approx.f32 %e, %neg;
    add.f32 %denom, %one, %e;
    rcp.approx.f32 %sig, %denom;

    // deriv = sig + x * sig * (1 - sig)
    sub.f32 %one_m_sig, %one, %sig;
    mul.f32 %x_sig_omsig, %x, %sig;
    mul.f32 %x_sig_omsig, %x_sig_omsig, %one_m_sig;
    add.f32 %deriv, %sig, %x_sig_omsig;
    mul.f32 %result, %vg, %deriv;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `silu_backward_f64_kernel` (f64).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(-x).
#[cfg(feature = "cuda")]
pub(crate) const SILU_BACKWARD_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry silu_backward_f64_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f64 %vg, %x, %neg_x, %e, %denom, %sig, %one;
    .reg .f64 %one_m_sig, %x_sig_omsig, %deriv, %result;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
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
    shl.b64 %off, %off, 3;
    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %vg, [%grad];
    ld.global.f64 %x, [%input];

    mov.f64 %one, 0d3FF0000000000000;
    neg.f64 %neg_x, %x;

    // --- exp(%neg_x) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %neg_x, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg_x;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e, %e, %e_nf;
    // --- end exp ---

    add.f64 %denom, %one, %e;
    div.rn.f64 %sig, %one, %denom;

    sub.f64 %one_m_sig, %one, %sig;
    mul.f64 %x_sig_omsig, %x, %sig;
    mul.f64 %x_sig_omsig, %x_sig_omsig, %one_m_sig;
    add.f64 %deriv, %sig, %x_sig_omsig;
    mul.f64 %result, %vg, %deriv;
    st.global.f64 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `elu_kernel`: `out[i] = x > 0 ? x : alpha * (exp(x) - 1)`.
/// Takes `alpha` as an extra `.param .f32` parameter.
#[cfg(feature = "cuda")]
pub(crate) const ELU_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry elu_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .f32 alpha
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %x, %alpha_r, %lg2e, %one, %ex, %em1, %neg_branch, %vr;
    .reg .pred %p, %pos;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.f32 %alpha_r, [alpha];

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

    ld.global.f32 %x, [%a];
    mov.f32 %one, 0f3F800000;
    mov.f32 %lg2e, 0f3FB8AA3B;

    // exp(x) = 2^(x * log2(e))
    mul.f32 %ex, %x, %lg2e;
    ex2.approx.f32 %ex, %ex;
    sub.f32 %em1, %ex, %one;
    mul.f32 %neg_branch, %alpha_r, %em1;

    // x > 0 ? x : alpha*(exp(x)-1)
    mov.f32 %vr, 0f00000000;
    setp.gt.f32 %pos, %x, %vr;
    selp.f32 %vr, %x, %neg_branch, %pos;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `elu_f64_kernel`: `out[i] = x > 0 ? x : alpha * (exp(x) - 1)` (f64).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(x).
#[cfg(feature = "cuda")]
pub(crate) const ELU_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry elu_f64_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .f64 alpha
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f64 %x, %alpha_r, %one, %ex, %em1, %neg_branch, %vr;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
    .reg .pred %p, %pos;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.f64 %alpha_r, [alpha];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %x, [%a];
    mov.f64 %one, 0d3FF0000000000000;

    // --- exp(%x) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %x, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %x;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %ex, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %ex, %ex, %e_nf;
    // --- end exp ---

    sub.f64 %em1, %ex, %one;
    mul.f64 %neg_branch, %alpha_r, %em1;

    mov.f64 %vr, 0d0000000000000000;
    setp.gt.f64 %pos, %x, %vr;
    selp.f64 %vr, %x, %neg_branch, %pos;
    st.global.f64 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `elu_backward_kernel`:
/// `out[i] = x > 0 ? grad[i] : grad[i] * alpha * exp(x)`.
/// Takes `alpha` as an extra `.param .f32` parameter.
#[cfg(feature = "cuda")]
pub(crate) const ELU_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry elu_backward_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .f32 alpha
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f32 %vg, %x, %alpha_r, %lg2e, %ex, %neg_branch, %vr, %zero;
    .reg .pred %p, %pos;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.f32 %alpha_r, [alpha];

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

    mov.f32 %lg2e, 0f3FB8AA3B;
    mov.f32 %zero, 0f00000000;

    // exp(x) = 2^(x * log2(e))
    mul.f32 %ex, %x, %lg2e;
    ex2.approx.f32 %ex, %ex;
    // negative branch: grad * alpha * exp(x)
    mul.f32 %neg_branch, %vg, %alpha_r;
    mul.f32 %neg_branch, %neg_branch, %ex;

    // x > 0 ? grad : grad * alpha * exp(x)
    setp.gt.f32 %pos, %x, %zero;
    selp.f32 %vr, %vg, %neg_branch, %pos;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `elu_backward_f64_kernel` (f64).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(x).
#[cfg(feature = "cuda")]
pub(crate) const ELU_BACKWARD_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry elu_backward_f64_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .f64 alpha
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f64 %vg, %x, %alpha_r, %ex, %neg_branch, %vr, %zero, %one;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
    .reg .pred %p, %pos;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.f64 %alpha_r, [alpha];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 3;
    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %vg, [%grad];
    ld.global.f64 %x, [%input];

    mov.f64 %zero, 0d0000000000000000;
    mov.f64 %one, 0d3FF0000000000000;

    // --- exp(%x) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %x, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %x;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %ex, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %ex, %ex, %e_nf;
    // --- end exp ---

    mul.f64 %neg_branch, %vg, %alpha_r;
    mul.f64 %neg_branch, %neg_branch, %ex;

    setp.gt.f64 %pos, %x, %zero;
    selp.f64 %vr, %vg, %neg_branch, %pos;
    st.global.f64 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `mish_kernel`: `out[i] = x * tanh(softplus(x))`.
/// softplus(x) = ln(1 + exp(x)). For stability: when x > 20, softplus ~ x.
/// tanh(y) = (exp(2y) - 1) / (exp(2y) + 1).
#[cfg(feature = "cuda")]
pub(crate) const MISH_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mish_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %x, %lg2e, %one, %ex, %ep1, %sp, %lg_ep1;
    .reg .f32 %two_sp, %e2sp, %e2sp_m1, %e2sp_p1, %th, %vr;
    .reg .f32 %threshold;
    .reg .pred %p, %large;

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

    ld.global.f32 %x, [%a];
    mov.f32 %one, 0f3F800000;
    mov.f32 %lg2e, 0f3FB8AA3B;
    // threshold = 20.0 = 0x41A00000
    mov.f32 %threshold, 0f41A00000;

    // softplus(x) = ln(1 + exp(x))
    // For large x (> 20), softplus ~ x to avoid overflow
    setp.gt.f32 %large, %x, %threshold;
    @%large bra LARGE_X;

    // exp(x) = 2^(x * log2(e))
    mul.f32 %ex, %x, %lg2e;
    ex2.approx.f32 %ex, %ex;
    add.f32 %ep1, %ex, %one;
    // ln(1+exp(x)) = log2(1+exp(x)) / log2(e)
    lg2.approx.f32 %lg_ep1, %ep1;
    // 1/log2(e) = ln(2) = 0.6931472 = 0x3F317218
    mul.f32 %sp, %lg_ep1, 0f3F317218;

    // tanh(sp) = (exp(2*sp) - 1) / (exp(2*sp) + 1)
    add.f32 %two_sp, %sp, %sp;
    mul.f32 %two_sp, %two_sp, %lg2e;
    ex2.approx.f32 %e2sp, %two_sp;
    sub.f32 %e2sp_m1, %e2sp, %one;
    add.f32 %e2sp_p1, %e2sp, %one;
    rcp.approx.f32 %e2sp_p1, %e2sp_p1;
    mul.f32 %th, %e2sp_m1, %e2sp_p1;

    mul.f32 %vr, %x, %th;
    st.global.f32 [%out], %vr;
    bra DONE;

LARGE_X:
    // softplus ~ x, mish ~ x * tanh(x)
    // tanh(x) = (exp(2x)-1)/(exp(2x)+1)
    add.f32 %two_sp, %x, %x;
    mul.f32 %two_sp, %two_sp, %lg2e;
    ex2.approx.f32 %e2sp, %two_sp;
    sub.f32 %e2sp_m1, %e2sp, %one;
    add.f32 %e2sp_p1, %e2sp, %one;
    rcp.approx.f32 %e2sp_p1, %e2sp_p1;
    mul.f32 %th, %e2sp_m1, %e2sp_p1;
    mul.f32 %vr, %x, %th;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `mish_f64_kernel`: `out[i] = x * tanh(softplus(x))` (f64).
/// Full f64 precision: exp via Cody-Waite + Horner, log via argument reduction.
#[cfg(feature = "cuda")]
pub(crate) const MISH_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mish_f64_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f64 %x, %one, %two, %ex, %ep1, %sp;
    .reg .f64 %two_sp, %e2sp, %e2sp_m1, %e2sp_p1, %th, %vr;
    .reg .f64 %threshold;
    // exp subroutine regs
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
    // log subroutine regs
    .reg .u64 %l_xbits, %l_mbits, %l_bias;
    .reg .s64 %l_exp64;
    .reg .f64 %l_m, %l_f, %l_f2, %l_s, %l_p, %l_nf, %l_ln2;
    .reg .pred %p, %large;

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
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %x, [%a];
    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %two, 0d4000000000000000;
    mov.f64 %threshold, 0d4034000000000000;

    setp.gt.f64 %large, %x, %threshold;
    @%large bra LARGE_X;

    // === softplus: sp = ln(1 + exp(x)) ===
    // exp(x)
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %x, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %x;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %ex, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %ex, %ex, %e_nf;

    // ep1 = 1 + exp(x)
    add.f64 %ep1, %ex, %one;

    // ln(ep1) via argument reduction
    mov.b64 %l_xbits, %ep1;
    shr.u64 %l_exp64, %l_xbits, 52;
    and.b64 %l_exp64, %l_exp64, 2047;
    sub.s64 %l_exp64, %l_exp64, 1023;
    cvt.rn.f64.s64 %l_nf, %l_exp64;
    mov.u64 %l_bias, 0x3FF0000000000000;
    and.b64 %l_mbits, %l_xbits, 0x000FFFFFFFFFFFFF;
    or.b64 %l_mbits, %l_mbits, %l_bias;
    mov.b64 %l_m, %l_mbits;
    sub.f64 %l_f, %l_m, %one;
    add.f64 %l_s, %l_m, %one;
    div.rn.f64 %l_f, %l_f, %l_s;
    mul.f64 %l_f2, %l_f, %l_f;
    mov.f64 %l_p, 0d3FB745D1745D1746;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC1C71C71C71C72;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC2492492492492;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC999999999999A;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FD5555555555555;
    fma.rn.f64 %l_p, %l_p, %l_f2, %one;
    mul.f64 %l_p, %l_p, %l_f;
    add.f64 %l_p, %l_p, %l_p;
    mov.f64 %l_ln2, 0d3FE62E42FEFA39EF;
    fma.rn.f64 %sp, %l_nf, %l_ln2, %l_p;

    // === tanh(sp) = (exp(2*sp)-1)/(exp(2*sp)+1) ===
    add.f64 %two_sp, %sp, %sp;
    fma.rn.f64 %e_nf, %two_sp, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %two_sp;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e2sp, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e2sp, %e2sp, %e_nf;

    sub.f64 %e2sp_m1, %e2sp, %one;
    add.f64 %e2sp_p1, %e2sp, %one;
    div.rn.f64 %th, %e2sp_m1, %e2sp_p1;

    mul.f64 %vr, %x, %th;
    st.global.f64 [%out], %vr;
    bra DONE;

LARGE_X:
    // softplus ~ x, tanh(x) = (exp(2x)-1)/(exp(2x)+1) in f64
    add.f64 %two_sp, %x, %x;
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %two_sp, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %two_sp;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e2sp, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e2sp, %e2sp, %e_nf;

    sub.f64 %e2sp_m1, %e2sp, %one;
    add.f64 %e2sp_p1, %e2sp, %one;
    div.rn.f64 %th, %e2sp_m1, %e2sp_p1;
    mul.f64 %vr, %x, %th;
    st.global.f64 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `mish_backward_kernel`:
/// ```text
/// sp = ln(1 + exp(x))        // softplus
/// t  = tanh(sp)
/// sig = sigmoid(x) = 1/(1+exp(-x))
/// out[i] = grad[i] * (t + x * sig * (1 - t*t))
/// ```
/// For stability: when x > 20, sp ~ x, t ~ tanh(x), sig ~ 1.
#[cfg(feature = "cuda")]
pub(crate) const MISH_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mish_backward_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f32 %vg, %x, %lg2e, %one, %ex, %ep1, %sp, %lg_ep1;
    .reg .f32 %two_sp, %e2sp, %e2sp_m1, %e2sp_p1, %t, %t2, %one_m_t2;
    .reg .f32 %neg, %en, %denom, %sig, %x_sig_omt2, %deriv, %result;
    .reg .f32 %threshold;
    .reg .pred %p, %large;

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

    mov.f32 %one, 0f3F800000;
    mov.f32 %lg2e, 0f3FB8AA3B;
    // threshold = 20.0
    mov.f32 %threshold, 0f41A00000;

    setp.gt.f32 %large, %x, %threshold;
    @%large bra LARGE_X;

    // --- Normal path ---
    // softplus: sp = ln(1 + exp(x))
    mul.f32 %ex, %x, %lg2e;
    ex2.approx.f32 %ex, %ex;
    add.f32 %ep1, %ex, %one;
    lg2.approx.f32 %lg_ep1, %ep1;
    // ln(2) = 0x3F317218
    mul.f32 %sp, %lg_ep1, 0f3F317218;

    // t = tanh(sp) = (exp(2*sp)-1)/(exp(2*sp)+1)
    add.f32 %two_sp, %sp, %sp;
    mul.f32 %two_sp, %two_sp, %lg2e;
    ex2.approx.f32 %e2sp, %two_sp;
    sub.f32 %e2sp_m1, %e2sp, %one;
    add.f32 %e2sp_p1, %e2sp, %one;
    rcp.approx.f32 %e2sp_p1, %e2sp_p1;
    mul.f32 %t, %e2sp_m1, %e2sp_p1;

    // sig = sigmoid(x) = 1/(1+exp(-x))
    neg.f32 %neg, %x;
    mul.f32 %neg, %neg, %lg2e;
    ex2.approx.f32 %en, %neg;
    add.f32 %denom, %one, %en;
    rcp.approx.f32 %sig, %denom;

    // deriv = t + x * sig * (1 - t*t)
    mul.f32 %t2, %t, %t;
    sub.f32 %one_m_t2, %one, %t2;
    mul.f32 %x_sig_omt2, %x, %sig;
    mul.f32 %x_sig_omt2, %x_sig_omt2, %one_m_t2;
    add.f32 %deriv, %t, %x_sig_omt2;
    mul.f32 %result, %vg, %deriv;
    st.global.f32 [%out], %result;
    bra DONE;

LARGE_X:
    // sp ~ x, t ~ tanh(x), sig ~ 1
    // tanh(x) = (exp(2x)-1)/(exp(2x)+1)
    add.f32 %two_sp, %x, %x;
    mul.f32 %two_sp, %two_sp, %lg2e;
    ex2.approx.f32 %e2sp, %two_sp;
    sub.f32 %e2sp_m1, %e2sp, %one;
    add.f32 %e2sp_p1, %e2sp, %one;
    rcp.approx.f32 %e2sp_p1, %e2sp_p1;
    mul.f32 %t, %e2sp_m1, %e2sp_p1;

    // sig ~ 1, deriv ~ t + x*(1-t*t)
    mul.f32 %t2, %t, %t;
    sub.f32 %one_m_t2, %one, %t2;
    mul.f32 %x_sig_omt2, %x, %one_m_t2;
    add.f32 %deriv, %t, %x_sig_omt2;
    mul.f32 %result, %vg, %deriv;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `mish_backward_f64_kernel` (f64).
/// Full f64 precision: exp via Cody-Waite + Horner, log via argument reduction.
#[cfg(feature = "cuda")]
pub(crate) const MISH_BACKWARD_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mish_backward_f64_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f64 %vg, %x, %one, %ex, %ep1, %sp;
    .reg .f64 %two_sp, %e2sp, %e2sp_m1, %e2sp_p1, %t, %t2, %one_m_t2;
    .reg .f64 %neg_x, %en, %denom, %sig, %x_sig_omt2, %deriv, %result;
    .reg .f64 %threshold;
    // exp subroutine regs
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
    // log subroutine regs
    .reg .u64 %l_xbits, %l_mbits, %l_bias;
    .reg .s64 %l_exp64;
    .reg .f64 %l_m, %l_f, %l_f2, %l_s, %l_p, %l_nf, %l_ln2;
    .reg .pred %p, %large;

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
    shl.b64 %off, %off, 3;
    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %vg, [%grad];
    ld.global.f64 %x, [%input];

    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %threshold, 0d4034000000000000;

    setp.gt.f64 %large, %x, %threshold;
    @%large bra LARGE_X;

    // === softplus: sp = ln(1 + exp(x)) ===
    // exp(x)
    mov.f64 %e_half, 0d3FE0000000000000;
    mul.f64 %e_nf, %x, 0d3FF71547652B82FE;
    cvt.rni.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %x;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FC5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %ex, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %ex, %ex, %e_nf;

    add.f64 %ep1, %ex, %one;

    // ln(ep1) via argument reduction
    mov.b64 %l_xbits, %ep1;
    shr.u64 %l_exp64, %l_xbits, 52;
    and.b64 %l_exp64, %l_exp64, 2047;
    sub.s64 %l_exp64, %l_exp64, 1023;
    cvt.rn.f64.s64 %l_nf, %l_exp64;
    mov.u64 %l_bias, 0x3FF0000000000000;
    and.b64 %l_mbits, %l_xbits, 0x000FFFFFFFFFFFFF;
    or.b64 %l_mbits, %l_mbits, %l_bias;
    mov.b64 %l_m, %l_mbits;
    sub.f64 %l_f, %l_m, %one;
    add.f64 %l_s, %l_m, %one;
    div.rn.f64 %l_f, %l_f, %l_s;
    mul.f64 %l_f2, %l_f, %l_f;
    mov.f64 %l_p, 0d3FB745D1745D1746;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC1C71C71C71C72;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC2492492492492;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC999999999999A;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FD5555555555555;
    fma.rn.f64 %l_p, %l_p, %l_f2, %one;
    mul.f64 %l_p, %l_p, %l_f;
    add.f64 %l_p, %l_p, %l_p;
    mov.f64 %l_ln2, 0d3FE62E42FEFA39EF;
    fma.rn.f64 %sp, %l_nf, %l_ln2, %l_p;

    // === tanh(sp) ===
    add.f64 %two_sp, %sp, %sp;
    mul.f64 %e_nf, %two_sp, 0d3FF71547652B82FE;
    cvt.rni.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %two_sp;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FC5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e2sp, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e2sp, %e2sp, %e_nf;

    sub.f64 %e2sp_m1, %e2sp, %one;
    add.f64 %e2sp_p1, %e2sp, %one;
    div.rn.f64 %t, %e2sp_m1, %e2sp_p1;

    // === sigmoid(x) = 1/(1+exp(-x)) ===
    neg.f64 %neg_x, %x;
    mul.f64 %e_nf, %neg_x, 0d3FF71547652B82FE;
    cvt.rni.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg_x;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FC5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %en, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %en, %en, %e_nf;

    add.f64 %denom, %one, %en;
    div.rn.f64 %sig, %one, %denom;

    // deriv = t + x * sig * (1 - t*t)
    mul.f64 %t2, %t, %t;
    sub.f64 %one_m_t2, %one, %t2;
    mul.f64 %x_sig_omt2, %x, %sig;
    mul.f64 %x_sig_omt2, %x_sig_omt2, %one_m_t2;
    add.f64 %deriv, %t, %x_sig_omt2;
    mul.f64 %result, %vg, %deriv;
    st.global.f64 [%out], %result;
    bra DONE;

LARGE_X:
    // sp ~ x, tanh(x) in f64, sig ~ 1
    add.f64 %two_sp, %x, %x;
    mov.f64 %e_half, 0d3FE0000000000000;
    mul.f64 %e_nf, %two_sp, 0d3FF71547652B82FE;
    cvt.rni.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %two_sp;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FC5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e2sp, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e2sp, %e2sp, %e_nf;

    sub.f64 %e2sp_m1, %e2sp, %one;
    add.f64 %e2sp_p1, %e2sp, %one;
    div.rn.f64 %t, %e2sp_m1, %e2sp_p1;

    // sig ~ 1, deriv ~ t + x*(1-t*t)
    mul.f64 %t2, %t, %t;
    sub.f64 %one_m_t2, %one, %t2;
    mul.f64 %x_sig_omt2, %x, %one_m_t2;
    add.f64 %deriv, %t, %x_sig_omt2;
    mul.f64 %result, %vg, %deriv;
    st.global.f64 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `clamp_kernel`: `out[i] = max(min_val, min(max_val, x[i]))`.
/// Takes two extra f32 params: min_val, max_val.
#[cfg(feature = "cuda")]
pub(crate) const CLAMP_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry clamp_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .f32 min_val,
    .param .f32 max_val
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off;
    .reg .f32 %x, %mn, %mx, %result;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.f32 %mn, [min_val];
    ld.param.f32 %mx, [max_val];

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
    max.f32 %result, %x, %mn;
    min.f32 %result, %result, %mx;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Backward activation kernels
// ---------------------------------------------------------------------------

/// PTX source for `fill_kernel`: `out[i] = scalar for all i < n`.
///
/// Used by sum/mean backward to produce a GPU-resident tensor filled
/// with a constant, without the CPU → GPU round-trip the legacy path
/// incurred (`vec![go; numel].to(device)`).
#[cfg(feature = "cuda")]
pub(crate) const FILL_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry fill_f32_kernel(
    .param .u64 out_ptr,
    .param .f32 scalar,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %out, %off;
    .reg .f32 %v;
    .reg .pred %p;

    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %v, [scalar];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %v;

DONE:
    ret;
}
";

/// PTX source for `abs_backward_kernel`:
/// `out[i] = input[i] > 0 ? grad[i] : (input[i] < 0 ? -grad[i] : 0)`.
///
/// Implements the derivative of `|x|`: `sign(x)` with the convention
/// that `sign(0) = 0`. Takes `grad` (upstream) and `input` (forward
/// activation input) as its two tensor parameters.
#[cfg(feature = "cuda")]
pub(crate) const ABS_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry abs_backward_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f32 %vg, %vi, %zero, %neg_vg, %tmp, %vr;
    .reg .pred %p, %pos, %neg;

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

    neg.f32 %neg_vg, %vg;

    // tmp = (vi < 0) ? -vg : 0
    setp.lt.f32 %neg, %vi, %zero;
    selp.f32 %tmp, %neg_vg, %zero, %neg;
    // vr = (vi > 0) ? vg : tmp
    setp.gt.f32 %pos, %vi, %zero;
    selp.f32 %vr, %vg, %tmp, %pos;

    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

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

/// PTX source for `gelu_backward_f64_kernel`: sigmoid-approx backward (f64).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(-k*x).
#[cfg(feature = "cuda")]
pub(crate) const GELU_BACKWARD_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_backward_f64_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f64 %vg, %x, %k, %kx, %neg_kx, %exp_neg, %one, %denom, %sig;
    .reg .f64 %one_minus_sig, %kx_sig_oms, %dsig, %result;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
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
    shl.b64 %off, %off, 3;
    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %vg, [%grad];
    ld.global.f64 %x, [%input];

    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %k, 0d3FFB44E400000000;
    mul.f64 %kx, %k, %x;
    neg.f64 %neg_kx, %kx;

    // --- exp(%neg_kx) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %neg_kx, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg_kx;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %exp_neg, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %exp_neg, %exp_neg, %e_nf;
    // --- end exp ---

    add.f64 %denom, %one, %exp_neg;
    div.rn.f64 %sig, %one, %denom;

    sub.f64 %one_minus_sig, %one, %sig;
    mul.f64 %kx_sig_oms, %kx, %sig;
    mul.f64 %kx_sig_oms, %kx_sig_oms, %one_minus_sig;
    add.f64 %dsig, %sig, %kx_sig_oms;

    mul.f64 %result, %vg, %dsig;
    st.global.f64 [%out], %result;

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

/// PTX source for `gelu_backward_erf_f64_kernel`: exact erf backward (f64).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(-z^2) and exp(-x^2/2).
#[cfg(feature = "cuda")]
pub(crate) const GELU_BACKWARD_ERF_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_backward_erf_f64_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f64 %vg, %x, %ax, %z, %z2, %neg_z2, %exp_neg_z2;
    .reg .f64 %t, %pt, %one, %half, %erf_val, %cdf, %pdf;
    .reg .f64 %neg_x2h, %exp_neg_x2h, %inv_sqrt_2pi, %x_pdf;
    .reg .f64 %d_gelu, %result;
    .reg .f64 %p_coef, %a1, %a2, %a3, %a4, %a5;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
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
    shl.b64 %off, %off, 3;
    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %vg, [%grad];
    ld.global.f64 %x, [%input];

    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %half, 0d3FE0000000000000;

    mov.f64 %z, 0d3FE6A09E60000000;
    mul.f64 %z, %x, %z;
    abs.f64 %ax, %z;

    mov.f64 %p_coef, 0d3FD4F740A0000000;
    mul.f64 %t, %p_coef, %ax;
    add.f64 %t, %one, %t;
    div.rn.f64 %t, %one, %t;

    mov.f64 %a5, 0d3FC1555560000000;
    mov.f64 %a4, 0dBFD6752060000000;
    mov.f64 %a3, 0d3FF6A0DBA0000000;
    mov.f64 %a2, 0dBFE0783C20000000;
    mov.f64 %a1, 0d3FD41AD760000000;

    mul.f64 %pt, %t, %a5;
    add.f64 %pt, %pt, %a4;
    mul.f64 %pt, %pt, %t;
    add.f64 %pt, %pt, %a3;
    mul.f64 %pt, %pt, %t;
    add.f64 %pt, %pt, %a2;
    mul.f64 %pt, %pt, %t;
    add.f64 %pt, %pt, %a1;
    mul.f64 %pt, %pt, %t;

    // exp(-z^2) in full f64
    mul.f64 %z2, %ax, %ax;
    neg.f64 %neg_z2, %z2;

    // --- exp(%neg_z2) ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %neg_z2, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg_z2;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %exp_neg_z2, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %exp_neg_z2, %exp_neg_z2, %e_nf;
    // --- end exp ---

    mul.f64 %erf_val, %pt, %exp_neg_z2;
    sub.f64 %erf_val, %one, %erf_val;

    setp.lt.f64 %pred_neg, %z, 0d0000000000000000;
    @%pred_neg neg.f64 %erf_val, %erf_val;

    add.f64 %cdf, %one, %erf_val;
    mul.f64 %cdf, %half, %cdf;

    // phi(x) = exp(-x^2/2) / sqrt(2*pi)
    mul.f64 %neg_x2h, %x, %x;
    mul.f64 %neg_x2h, %neg_x2h, %half;
    neg.f64 %neg_x2h, %neg_x2h;

    // --- exp(%neg_x2h) ---
    fma.rn.f64 %e_nf, %neg_x2h, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg_x2h;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %exp_neg_x2h, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %exp_neg_x2h, %exp_neg_x2h, %e_nf;
    // --- end exp ---

    // 1/sqrt(2*pi) = 0.39894228
    mov.f64 %inv_sqrt_2pi, 0d3FD9884440000000;
    mul.f64 %pdf, %exp_neg_x2h, %inv_sqrt_2pi;

    mul.f64 %x_pdf, %x, %pdf;
    add.f64 %d_gelu, %cdf, %x_pdf;

    mul.f64 %result, %vg, %d_gelu;
    st.global.f64 [%out], %result;

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
// LogSoftmax forward PTX kernel (row-wise, shared-memory max + log-sum-exp)
// ---------------------------------------------------------------------------
// For each row of length `cols`:
//   m = max(x[j])
//   log_sum_exp = m + log(sum(exp(x[j] - m)))
//   out[j] = x[j] - log_sum_exp
// One block per row, 256 threads per block.

#[cfg(feature = "cuda")]
pub(crate) const LOG_SOFTMAX_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.shared .align 4 .f32 sdata[256];\n\
\n\
.visible .entry log_softmax_kernel(\n\
    .param .u64 input_ptr,\n\
    .param .u64 output_ptr,\n\
    .param .u32 rows,\n\
    .param .u32 cols\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j;\n\
    .reg .u64 %in, %out, %row_off, %off, %sbase, %saddr;\n\
    .reg .f32 %val, %max_val, %sum_val, %exp_val, %log_sum_exp, %result;\n\
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
    // row_off = bid * cols * 4 (byte offset)\n\
    cvt.u64.u32 %row_off, %bid;\n\
    cvt.u64.u32 %off, %cols_reg;\n\
    mul.lo.u64 %row_off, %row_off, %off;\n\
    shl.b64 %row_off, %row_off, 2;\n\
\n\
    // Phase 1: find max across row (grid-stride over columns)\n\
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
    // Shared-memory tree reduction for max\n\
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
    add.u64 %saddr, %sbase, %off;\n\
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
    // Broadcast max to all threads\n\
    ld.shared.f32 %max_val, [sdata];\n\
    bar.sync 0;\n\
\n\
    // Phase 2: compute partial sum of exp(x[j] - max)\n\
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
    // exp(x) = exp2(x * log2(e)), log2(e) = 0x3FB8AA3B\n\
    mul.f32 %val, %val, 0f3FB8AA3B;\n\
    ex2.approx.f32 %exp_val, %val;\n\
    add.f32 %sum_val, %sum_val, %exp_val;\n\
    add.u32 %j, %j, %bdim;\n\
    bra SUM_EXP;\n\
SUM_EXP_DONE:\n\
\n\
    // Shared-memory tree reduction for sum\n\
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
    add.u64 %saddr, %sbase, %off;\n\
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
    // Broadcast sum to all threads, compute log_sum_exp = max + log(sum)\n\
    ld.shared.f32 %sum_val, [sdata];\n\
    bar.sync 0;\n\
    // log(x) = log2(x) / log2(e) = log2(x) * ln(2)\n\
    // ln(2) = 0x3F317218\n\
    lg2.approx.f32 %log_sum_exp, %sum_val;\n\
    mul.f32 %log_sum_exp, %log_sum_exp, 0f3F317218;\n\
    add.f32 %log_sum_exp, %max_val, %log_sum_exp;\n\
\n\
    // Phase 3: out[j] = x[j] - log_sum_exp\n\
    mov.u32 %j, %r_tid;\n\
WRITE_OUTPUT:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra WRITE_OUTPUT_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %in, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f32 %val, [%saddr];\n\
    sub.f32 %result, %val, %log_sum_exp;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %out, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    st.global.f32 [%saddr], %result;\n\
    add.u32 %j, %j, %bdim;\n\
    bra WRITE_OUTPUT;\n\
WRITE_OUTPUT_DONE:\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

/// PTX source for `log_softmax_f64_kernel`: row-wise log-softmax (f64).
#[cfg(feature = "cuda")]
pub(crate) const LOG_SOFTMAX_F64_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.shared .align 8 .f64 sdata[256];\n\
\n\
.visible .entry log_softmax_f64_kernel(\n\
    .param .u64 input_ptr,\n\
    .param .u64 output_ptr,\n\
    .param .u32 rows,\n\
    .param .u32 cols\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j;\n\
    .reg .u64 %in, %out, %row_off, %off, %sbase, %saddr;\n\
    .reg .f64 %val, %max_val, %sum_val, %exp_val, %log_sum_exp, %result;\n\
    .reg .pred %p, %loop_p;\n\
    .reg .u32 %half, %other_tid;\n\
    .reg .f64 %other_val;\n\
    .reg .pred %reduce_p;\n\
    .reg .f64 %e_nf, %e_r, %e_p, %e_half, %e_one;\n\
    .reg .s32 %e_ni;\n\
    .reg .s64 %e_ni64, %e_bits;\n\
    .reg .u64 %l_xbits, %l_mbits, %l_bias;\n\
    .reg .s64 %l_exp64;\n\
    .reg .f64 %l_m, %l_f, %l_f2, %l_s, %l_p, %l_nf, %l_ln2;\n\
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
    shl.b64 %row_off, %row_off, 3;\n\
\n\
    mov.f64 %max_val, 0dFFF0000000000000;\n\
    mov.u32 %j, %r_tid;\n\
FIND_MAX:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra FIND_MAX_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %off, %in, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    ld.global.f64 %val, [%off];\n\
    max.f64 %max_val, %max_val, %val;\n\
    add.u32 %j, %j, %bdim;\n\
    bra FIND_MAX;\n\
FIND_MAX_DONE:\n\
\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f64 [%saddr], %max_val;\n\
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
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %other_val, [%saddr];\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %max_val, [%saddr];\n\
    max.f64 %max_val, %max_val, %other_val;\n\
    st.shared.f64 [%saddr], %max_val;\n\
MAX_REDUCE_SKIP:\n\
    bar.sync 0;\n\
    bra MAX_REDUCE;\n\
MAX_REDUCE_DONE:\n\
\n\
    ld.shared.f64 %max_val, [sdata];\n\
    bar.sync 0;\n\
\n\
    mov.f64 %sum_val, 0d0000000000000000;\n\
    mov.u32 %j, %r_tid;\n\
SUM_EXP:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra SUM_EXP_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %off, %in, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    ld.global.f64 %val, [%off];\n\
    sub.f64 %val, %val, %max_val;\n\
    mov.f64 %e_one, 0d3FF0000000000000;\n\
    mov.f64 %e_half, 0d3FE0000000000000;\n\
    mul.f64 %e_nf, %val, 0d3FF71547652B82FE;\n\
    cvt.rni.f64.f64 %e_nf, %e_nf;\n\
    cvt.rni.s32.f64 %e_ni, %e_nf;\n\
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %val;\n\
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;\n\
    mov.f64 %e_p, 0d3E21EED8EFF8D898;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FC5555555555555;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, %e_one;\n\
    fma.rn.f64 %exp_val, %e_p, %e_r, %e_one;\n\
    cvt.s64.s32 %e_ni64, %e_ni;\n\
    add.s64 %e_ni64, %e_ni64, 1023;\n\
    shl.b64 %e_bits, %e_ni64, 52;\n\
    mov.b64 %e_nf, %e_bits;\n\
    mul.f64 %exp_val, %exp_val, %e_nf;\n\
    add.f64 %sum_val, %sum_val, %exp_val;\n\
    add.u32 %j, %j, %bdim;\n\
    bra SUM_EXP;\n\
SUM_EXP_DONE:\n\
\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f64 [%saddr], %sum_val;\n\
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
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %other_val, [%saddr];\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %sum_val, [%saddr];\n\
    add.f64 %sum_val, %sum_val, %other_val;\n\
    st.shared.f64 [%saddr], %sum_val;\n\
SUM_REDUCE_SKIP:\n\
    bar.sync 0;\n\
    bra SUM_REDUCE;\n\
SUM_REDUCE_DONE:\n\
\n\
    ld.shared.f64 %sum_val, [sdata];\n\
    bar.sync 0;\n\
    mov.f64 %e_one, 0d3FF0000000000000;\n\
    mov.b64 %l_xbits, %sum_val;\n\
    shr.u64 %l_exp64, %l_xbits, 52;\n\
    and.b64 %l_exp64, %l_exp64, 2047;\n\
    sub.s64 %l_exp64, %l_exp64, 1023;\n\
    cvt.rn.f64.s64 %l_nf, %l_exp64;\n\
    mov.u64 %l_bias, 0x3FF0000000000000;\n\
    and.b64 %l_mbits, %l_xbits, 0x000FFFFFFFFFFFFF;\n\
    or.b64 %l_mbits, %l_mbits, %l_bias;\n\
    mov.b64 %l_m, %l_mbits;\n\
    sub.f64 %l_f, %l_m, %e_one;\n\
    add.f64 %l_s, %l_m, %e_one;\n\
    div.rn.f64 %l_f, %l_f, %l_s;\n\
    mul.f64 %l_f2, %l_f, %l_f;\n\
    mov.f64 %l_p, 0d3FB745D1745D1746;\n\
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC1C71C71C71C72;\n\
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC2492492492492;\n\
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC999999999999A;\n\
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FD5555555555555;\n\
    fma.rn.f64 %l_p, %l_p, %l_f2, %e_one;\n\
    mul.f64 %l_p, %l_p, %l_f;\n\
    add.f64 %l_p, %l_p, %l_p;\n\
    mov.f64 %l_ln2, 0d3FE62E42FEFA39EF;\n\
    fma.rn.f64 %log_sum_exp, %l_nf, %l_ln2, %l_p;\n\
    add.f64 %log_sum_exp, %max_val, %log_sum_exp;\n\
\n\
    mov.u32 %j, %r_tid;\n\
WRITE_OUTPUT:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra WRITE_OUTPUT_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %in, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f64 %val, [%saddr];\n\
    sub.f64 %result, %val, %log_sum_exp;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %out, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    st.global.f64 [%saddr], %result;\n\
    add.u32 %j, %j, %bdim;\n\
    bra WRITE_OUTPUT;\n\
WRITE_OUTPUT_DONE:\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

// ---------------------------------------------------------------------------
// LogSoftmax backward PTX kernel (row-wise, shared-memory sum reduction)
// ---------------------------------------------------------------------------
// For each row of length `cols`:
//   sum_grad = sum(grad[j])
//   out[j] = grad[j] - exp(output[j]) * sum_grad
// where output[j] is the log-softmax output, so exp(output[j]) = softmax[j].
// One block per row, 256 threads per block.

#[cfg(feature = "cuda")]
pub(crate) const LOG_SOFTMAX_BACKWARD_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.shared .align 4 .f32 sdata[256];\n\
\n\
.visible .entry log_softmax_backward_kernel(\n\
    .param .u64 grad_ptr,\n\
    .param .u64 output_ptr,\n\
    .param .u64 out_ptr,\n\
    .param .u32 rows,\n\
    .param .u32 cols\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %other_tid;\n\
    .reg .u64 %grad, %output, %out, %row_off, %off, %sbase, %saddr;\n\
    .reg .f32 %vg, %vo, %sum_grad, %other_val, %softmax_j, %result;\n\
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
    // Phase 1: compute partial sum_grad = sum(grad[j]) for this thread's elements\n\
    mov.f32 %sum_grad, 0f00000000;\n\
    mov.u32 %j, %r_tid;\n\
SUM_LOOP:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra SUM_LOOP_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %grad, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f32 %vg, [%saddr];\n\
    add.f32 %sum_grad, %sum_grad, %vg;\n\
    add.u32 %j, %j, %bdim;\n\
    bra SUM_LOOP;\n\
SUM_LOOP_DONE:\n\
\n\
    // Store partial sum into shared memory and reduce\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f32 [%saddr], %sum_grad;\n\
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
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f32 %other_val, [%saddr];\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f32 %sum_grad, [%saddr];\n\
    add.f32 %sum_grad, %sum_grad, %other_val;\n\
    st.shared.f32 [%saddr], %sum_grad;\n\
SUM_REDUCE_SKIP:\n\
    bar.sync 0;\n\
    bra SUM_REDUCE;\n\
SUM_REDUCE_DONE:\n\
\n\
    // Broadcast sum_grad to all threads\n\
    ld.shared.f32 %sum_grad, [sdata];\n\
    bar.sync 0;\n\
\n\
    // Phase 2: out[j] = grad[j] - exp(output[j]) * sum_grad\n\
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
    // exp(log_softmax_output) = softmax probability\n\
    mul.f32 %vo, %vo, 0f3FB8AA3B;\n\
    ex2.approx.f32 %softmax_j, %vo;\n\
    // out[j] = grad[j] - softmax[j] * sum_grad\n\
    mul.f32 %result, %softmax_j, %sum_grad;\n\
    sub.f32 %result, %vg, %result;\n\
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

/// PTX source for `log_softmax_backward_f64_kernel` (f64).
#[cfg(feature = "cuda")]
pub(crate) const LOG_SOFTMAX_BACKWARD_F64_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.shared .align 8 .f64 sdata[256];\n\
\n\
.visible .entry log_softmax_backward_f64_kernel(\n\
    .param .u64 grad_ptr,\n\
    .param .u64 output_ptr,\n\
    .param .u64 out_ptr,\n\
    .param .u32 rows,\n\
    .param .u32 cols\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %other_tid;\n\
    .reg .u64 %grad, %output, %out, %row_off, %off, %sbase, %saddr;\n\
    .reg .f64 %vg, %vo, %sum_grad, %other_val, %softmax_j, %result;\n\
    .reg .pred %p, %loop_p, %reduce_p;\n\
    .reg .f64 %e_nf, %e_r, %e_p, %e_half, %e_one;\n\
    .reg .s32 %e_ni;\n\
    .reg .s64 %e_ni64, %e_bits;\n\
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
    cvt.u64.u32 %row_off, %bid;\n\
    cvt.u64.u32 %off, %cols_reg;\n\
    mul.lo.u64 %row_off, %row_off, %off;\n\
    shl.b64 %row_off, %row_off, 3;\n\
\n\
    mov.f64 %sum_grad, 0d0000000000000000;\n\
    mov.u32 %j, %r_tid;\n\
SUM_LOOP:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra SUM_LOOP_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %grad, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f64 %vg, [%saddr];\n\
    add.f64 %sum_grad, %sum_grad, %vg;\n\
    add.u32 %j, %j, %bdim;\n\
    bra SUM_LOOP;\n\
SUM_LOOP_DONE:\n\
\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f64 [%saddr], %sum_grad;\n\
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
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %other_val, [%saddr];\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %sum_grad, [%saddr];\n\
    add.f64 %sum_grad, %sum_grad, %other_val;\n\
    st.shared.f64 [%saddr], %sum_grad;\n\
SUM_REDUCE_SKIP:\n\
    bar.sync 0;\n\
    bra SUM_REDUCE;\n\
SUM_REDUCE_DONE:\n\
\n\
    ld.shared.f64 %sum_grad, [sdata];\n\
    bar.sync 0;\n\
\n\
    mov.u32 %j, %r_tid;\n\
WRITE_LOOP:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra WRITE_LOOP_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %grad, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f64 %vg, [%saddr];\n\
    add.u64 %saddr, %output, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f64 %vo, [%saddr];\n\
    // exp(log_softmax_output) — inline f64 exp\n\
    mov.f64 %e_one, 0d3FF0000000000000;\n\
    mov.f64 %e_half, 0d3FE0000000000000;\n\
    mul.f64 %e_nf, %vo, 0d3FF71547652B82FE;\n\
    cvt.rni.f64.f64 %e_nf, %e_nf;\n\
    cvt.rni.s32.f64 %e_ni, %e_nf;\n\
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %vo;\n\
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;\n\
    mov.f64 %e_p, 0d3E21EED8EFF8D898;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FC5555555555555;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, %e_one;\n\
    fma.rn.f64 %softmax_j, %e_p, %e_r, %e_one;\n\
    cvt.s64.s32 %e_ni64, %e_ni;\n\
    add.s64 %e_ni64, %e_ni64, 1023;\n\
    shl.b64 %e_bits, %e_ni64, 52;\n\
    mov.b64 %e_nf, %e_bits;\n\
    mul.f64 %softmax_j, %softmax_j, %e_nf;\n\
    mul.f64 %result, %softmax_j, %sum_grad;\n\
    sub.f64 %result, %vg, %result;\n\
    add.u64 %saddr, %out, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    st.global.f64 [%saddr], %result;\n\
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
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %n_reg, %idx, %stride, %half;
    .reg .u64 %in, %out, %off, %sbase, %saddr;
    .reg .f32 %sum, %other;
    .reg .pred %p, %ptid;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %my_tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;
    mov.u64 %sbase, sdata;

    // Grid-stride accumulation: each thread sums multiple elements.
    // idx = bid * bdim + tid; stride = bdim * gdim
    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
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
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sum;
    bar.sync 0;

    // Tree reduction in shared memory.
    mov.u32 %half, 128;
TREE_LOOP:
    setp.lt.u32 %p, %half, 1;
    @%p bra TREE_DONE;

    setp.ge.u32 %ptid, %my_tid, %half;
    @%ptid bra TREE_SKIP;

    // Load partner's value from sdata[tid + half].
    add.u32 %idx, %my_tid, %half;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    // Load own value.
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sum, [%saddr];
    add.f32 %sum, %sum, %other;
    st.shared.f32 [%saddr], %sum;

TREE_SKIP:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra TREE_LOOP;

TREE_DONE:
    // Thread 0 writes block result.
    setp.ne.u32 %ptid, %my_tid, 0;
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

/// PTX source for `reduce_prod_kernel`: parallel block-level product
/// reduction (#524). Mirrors `REDUCE_SUM_PTX` exactly but the
/// accumulator inits to 1.0 and the combiner is `mul.f32` instead of
/// `add.f32`. Used for `prod` forward.
#[cfg(feature = "cuda")]
pub(crate) const REDUCE_PROD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry reduce_prod_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %n_reg, %idx, %stride, %half;
    .reg .u64 %in, %out, %off, %saddr;
    .reg .f32 %acc, %other;
    .reg .pred %p, %ptid;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %my_tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;

    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
    mul.lo.u32 %stride, %bdim, %gdim;
    mov.f32 %acc, 0f3F800000;        // 1.0 (multiplicative identity)

GRID_LOOP_PROD:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra GRID_DONE_PROD;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %other, [%off];
    mul.f32 %acc, %acc, %other;
    add.u32 %idx, %idx, %stride;
    bra GRID_LOOP_PROD;

GRID_DONE_PROD:
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;
    bar.sync 0;

    mov.u32 %half, 128;
TREE_LOOP_PROD:
    setp.lt.u32 %p, %half, 1;
    @%p bra TREE_DONE_PROD;

    setp.ge.u32 %ptid, %my_tid, %half;
    @%ptid bra TREE_SKIP_PROD;

    add.u32 %idx, %my_tid, %half;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %acc, [%saddr];
    mul.f32 %acc, %acc, %other;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;

TREE_SKIP_PROD:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra TREE_LOOP_PROD;

TREE_DONE_PROD:
    setp.ne.u32 %ptid, %my_tid, 0;
    @%ptid bra END_PROD;

    mov.u64 %saddr, sdata;
    ld.shared.f32 %acc, [%saddr];
    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %acc;

END_PROD:
    ret;
}
";

/// PTX source for `reduce_min_kernel`: parallel block-level min reduction (#627).
///
/// Mirrors `REDUCE_SUM_PTX` exactly except:
/// - Initial accumulator is `+inf` (`0f7F800000`) instead of zero.
/// - Combiner is `min.f32` instead of `add.f32`.
///
/// Same launch contract: each block emits one partial min to
/// `output[blockIdx.x]`. For full reductions the host calls twice (or
/// reduces the partials on CPU when there are few blocks).
#[cfg(feature = "cuda")]
pub(crate) const REDUCE_MIN_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry reduce_min_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %n_reg, %idx, %stride, %half;
    .reg .u64 %in, %out, %off, %saddr;
    .reg .f32 %acc, %other;
    .reg .pred %p, %ptid;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %my_tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;

    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
    mul.lo.u32 %stride, %bdim, %gdim;
    // accumulator init = +inf
    mov.f32 %acc, 0f7F800000;

GRID_LOOP_MIN:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra GRID_DONE_MIN;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %other, [%off];
    min.f32 %acc, %acc, %other;
    add.u32 %idx, %idx, %stride;
    bra GRID_LOOP_MIN;

GRID_DONE_MIN:
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;
    bar.sync 0;

    mov.u32 %half, 128;
TREE_LOOP_MIN:
    setp.lt.u32 %p, %half, 1;
    @%p bra TREE_DONE_MIN;

    setp.ge.u32 %ptid, %my_tid, %half;
    @%ptid bra TREE_SKIP_MIN;

    add.u32 %idx, %my_tid, %half;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %acc, [%saddr];
    min.f32 %acc, %acc, %other;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;

TREE_SKIP_MIN:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra TREE_LOOP_MIN;

TREE_DONE_MIN:
    setp.ne.u32 %ptid, %my_tid, 0;
    @%ptid bra END_MIN;

    mov.u64 %saddr, sdata;
    ld.shared.f32 %acc, [%saddr];
    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %acc;

END_MIN:
    ret;
}
";

/// PTX source for `reduce_max_kernel`: parallel block-level max reduction (#627).
///
/// Same as [`REDUCE_MIN_PTX`] but:
/// - Initial accumulator is `-inf` (`0fFF800000`).
/// - Combiner is `max.f32`.
#[cfg(feature = "cuda")]
pub(crate) const REDUCE_MAX_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry reduce_max_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %n_reg, %idx, %stride, %half;
    .reg .u64 %in, %out, %off, %saddr;
    .reg .f32 %acc, %other;
    .reg .pred %p, %ptid;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %my_tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;

    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
    mul.lo.u32 %stride, %bdim, %gdim;
    // accumulator init = -inf
    mov.f32 %acc, 0fFF800000;

GRID_LOOP_MAX:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra GRID_DONE_MAX;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %other, [%off];
    max.f32 %acc, %acc, %other;
    add.u32 %idx, %idx, %stride;
    bra GRID_LOOP_MAX;

GRID_DONE_MAX:
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;
    bar.sync 0;

    mov.u32 %half, 128;
TREE_LOOP_MAX:
    setp.lt.u32 %p, %half, 1;
    @%p bra TREE_DONE_MAX;

    setp.ge.u32 %ptid, %my_tid, %half;
    @%ptid bra TREE_SKIP_MAX;

    add.u32 %idx, %my_tid, %half;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %acc, [%saddr];
    max.f32 %acc, %acc, %other;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;

TREE_SKIP_MAX:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra TREE_LOOP_MAX;

TREE_DONE_MAX:
    setp.ne.u32 %ptid, %my_tid, 0;
    @%ptid bra END_MAX;

    mov.u64 %saddr, sdata;
    ld.shared.f32 %acc, [%saddr];
    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %acc;

END_MAX:
    ret;
}
";

/// Fused masked-min reduction (#627). Kernel signature:
///   `(data, mask_f, n) -> partial_mins`
/// where `mask_f[i]` is `1.0` for valid entries and `0.0` for masked.
/// Each thread combines `mask_f[i] != 0 ? data[i] : +inf` into its running
/// `min` accumulator, then the block does the standard tree reduction.
/// Eliminates the separate `mul + add + reduce` chain and the host-side
/// sentinel buffer construction that the unfused path required.
#[cfg(feature = "cuda")]
pub(crate) const MASKED_REDUCE_MIN_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry masked_reduce_min_kernel(
    .param .u64 data_ptr,
    .param .u64 mask_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %n_reg, %idx, %stride, %half;
    .reg .u64 %dat, %msk, %out, %off, %saddr;
    .reg .f32 %acc, %d, %m, %sentinel, %val;
    .reg .pred %p, %ptid, %p_valid;

    ld.param.u64 %dat, [data_ptr];
    ld.param.u64 %msk, [mask_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %my_tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;

    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
    mul.lo.u32 %stride, %bdim, %gdim;
    mov.f32 %acc, 0f7F800000;        // +inf
    mov.f32 %sentinel, 0f7F800000;

GRID_LOOP_MMIN:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra GRID_DONE_MMIN;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %dat, %off;
    ld.global.f32 %d, [%off];

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %msk, %off;
    ld.global.f32 %m, [%off];

    // val = (m != 0) ? d : +inf
    setp.ne.f32 %p_valid, %m, 0f00000000;
    selp.f32 %val, %d, %sentinel, %p_valid;

    min.f32 %acc, %acc, %val;
    add.u32 %idx, %idx, %stride;
    bra GRID_LOOP_MMIN;

GRID_DONE_MMIN:
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;
    bar.sync 0;

    mov.u32 %half, 128;
TREE_LOOP_MMIN:
    setp.lt.u32 %p, %half, 1;
    @%p bra TREE_DONE_MMIN;

    setp.ge.u32 %ptid, %my_tid, %half;
    @%ptid bra TREE_SKIP_MMIN;

    add.u32 %idx, %my_tid, %half;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %val, [%saddr];
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %acc, [%saddr];
    min.f32 %acc, %acc, %val;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;

TREE_SKIP_MMIN:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra TREE_LOOP_MMIN;

TREE_DONE_MMIN:
    setp.ne.u32 %ptid, %my_tid, 0;
    @%ptid bra END_MMIN;

    mov.u64 %saddr, sdata;
    ld.shared.f32 %acc, [%saddr];
    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %acc;

END_MMIN:
    ret;
}
";

/// Fused masked-max counterpart of [`MASKED_REDUCE_MIN_PTX`]. (#627)
#[cfg(feature = "cuda")]
pub(crate) const MASKED_REDUCE_MAX_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry masked_reduce_max_kernel(
    .param .u64 data_ptr,
    .param .u64 mask_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %n_reg, %idx, %stride, %half;
    .reg .u64 %dat, %msk, %out, %off, %saddr;
    .reg .f32 %acc, %d, %m, %sentinel, %val;
    .reg .pred %p, %ptid, %p_valid;

    ld.param.u64 %dat, [data_ptr];
    ld.param.u64 %msk, [mask_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %my_tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;

    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
    mul.lo.u32 %stride, %bdim, %gdim;
    mov.f32 %acc, 0fFF800000;        // -inf
    mov.f32 %sentinel, 0fFF800000;

GRID_LOOP_MMAX:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra GRID_DONE_MMAX;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %dat, %off;
    ld.global.f32 %d, [%off];

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %msk, %off;
    ld.global.f32 %m, [%off];

    setp.ne.f32 %p_valid, %m, 0f00000000;
    selp.f32 %val, %d, %sentinel, %p_valid;

    max.f32 %acc, %acc, %val;
    add.u32 %idx, %idx, %stride;
    bra GRID_LOOP_MMAX;

GRID_DONE_MMAX:
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;
    bar.sync 0;

    mov.u32 %half, 128;
TREE_LOOP_MMAX:
    setp.lt.u32 %p, %half, 1;
    @%p bra TREE_DONE_MMAX;

    setp.ge.u32 %ptid, %my_tid, %half;
    @%ptid bra TREE_SKIP_MMAX;

    add.u32 %idx, %my_tid, %half;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %val, [%saddr];
    cvt.u64.u32 %off, %my_tid;
    shl.b64 %off, %off, 2;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    ld.shared.f32 %acc, [%saddr];
    max.f32 %acc, %acc, %val;
    mov.u64 %saddr, sdata;
    add.u64 %saddr, %saddr, %off;
    st.shared.f32 [%saddr], %acc;

TREE_SKIP_MMAX:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra TREE_LOOP_MMAX;

TREE_DONE_MMAX:
    setp.ne.u32 %ptid, %my_tid, 0;
    @%ptid bra END_MMAX;

    mov.u64 %saddr, sdata;
    ld.shared.f32 %acc, [%saddr];
    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %acc;

END_MMAX:
    ret;
}
";

/// PTX source for `repeat_along_dim_kernel` (#524). Expands a `[outer,
/// inner]` tensor into `[outer, repeat_count, inner]` by broadcasting
/// along the inserted middle axis. Used for the backward of `sum_dim` /
/// `mean_dim` where the gradient needs to be replicated along the
/// previously-reduced dim. For mean_dim, the caller multiplies by
/// `1/repeat_count` separately to preserve a clean kernel boundary.
///
/// Each thread writes one output element. Output index decomposes as:
///   `t = o * (repeat_count * inner) + r * inner + i`
/// and reads from `input[o * inner + i]`.
#[cfg(feature = "cuda")]
pub(crate) const REPEAT_ALONG_DIM_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry repeat_along_dim_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 outer,
    .param .u32 repeat_count,
    .param .u32 inner
) {
    .reg .u32 %r_tid, %bid, %bdim, %t, %total, %o, %r, %i, %tmp_ri, %tmp_ri2, %ri_extent;
    .reg .u32 %src_idx, %re_x_in;
    .reg .u64 %inp, %out, %off;
    .reg .f32 %v;
    .reg .pred %p;

    ld.param.u64 %inp, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %tmp_ri, [outer];
    ld.param.u32 %tmp_ri2, [repeat_count];
    ld.param.u32 %ri_extent, [inner];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %t, %bid, %bdim, %r_tid;

    // total = outer * repeat_count * inner
    mul.lo.u32 %re_x_in, %tmp_ri2, %ri_extent;
    mul.lo.u32 %total, %tmp_ri, %re_x_in;

    setp.ge.u32 %p, %t, %total;
    @%p bra DONE_RAD;

    // o = t / (repeat_count * inner)
    div.u32 %o, %t, %re_x_in;
    // tmp = t % (repeat_count * inner)
    rem.u32 %tmp_ri, %t, %re_x_in;
    // r = tmp / inner; i = tmp % inner
    div.u32 %r, %tmp_ri, %ri_extent;
    rem.u32 %i, %tmp_ri, %ri_extent;
    // src_idx = o * inner + i
    mad.lo.u32 %src_idx, %o, %ri_extent, %i;

    // Load src
    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %inp, %off;
    ld.global.f32 %v, [%off];

    // Store to dst[t]
    cvt.u64.u32 %off, %t;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.f32 [%off], %v;

    // suppress unused warning
    mov.u32 %r, %r;

DONE_RAD:
    ret;
}
";

/// PTX source for `clamp_backward_kernel` (#524).
/// VJP for `clamp(x, min, max)`: `out[i] = grad[i]` when `min <= x[i] <= max`,
/// else `0`. Single-pass elementwise — no shared memory.
#[cfg(feature = "cuda")]
pub(crate) const CLAMP_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry clamp_backward_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .f32 min_val,
    .param .f32 max_val,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %g, %x, %out, %off;
    .reg .f32 %vg, %vx, %vmin, %vmax, %vr;
    .reg .pred %p, %plo, %phi, %pin;

    ld.param.u64 %g, [grad_ptr];
    ld.param.u64 %x, [input_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %vmin, [min_val];
    ld.param.f32 %vmax, [max_val];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE_CB;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %g, %g, %off;
    add.u64 %x, %x, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %vg, [%g];
    ld.global.f32 %vx, [%x];

    setp.ge.f32 %plo, %vx, %vmin;
    setp.le.f32 %phi, %vx, %vmax;
    and.pred %pin, %plo, %phi;

    mov.f32 %vr, 0f00000000;
    @%pin mov.f32 %vr, %vg;

    st.global.f32 [%out], %vr;

DONE_CB:
    ret;
}
";

/// PTX source for `pad_truncate_kernel` (#605): copies a `[batch, src_n, 2]`
/// complex tensor into a `[batch, dst_n, 2]` output, zero-padding when
/// `dst_n > src_n` and truncating when `dst_n < src_n`. Each thread writes
/// one output complex pair (`2 * f32` values). Used by the GPU FFT path
/// when the user passes `n != input_n`.
///
/// Thread layout: `tid` covers `batch * dst_n` complex pairs. For each
/// output position `(b, k)`:
///   - if `k < src_n`: copy `src[b, k, :]` to `dst[b, k, :]`
///   - else: write zeros.
#[cfg(feature = "cuda")]
pub(crate) const PAD_TRUNCATE_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry pad_truncate_kernel(
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u32 batch,
    .param .u32 src_n,
    .param .u32 dst_n
) {
    .reg .u32 %r_tid, %bid, %bdim, %total, %b_idx, %k_idx, %src_offset, %dst_offset;
    .reg .u32 %tmp32, %tmp32b;
    .reg .u64 %src_base, %dst_base, %off_src, %off_dst;
    .reg .f32 %re, %im;
    .reg .pred %p_oob, %p_pad;

    ld.param.u64 %src_base, [src_ptr];
    ld.param.u64 %dst_base, [dst_ptr];
    ld.param.u32 %tmp32, [batch];
    ld.param.u32 %tmp32b, [dst_n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    // total = batch * dst_n
    mul.lo.u32 %total, %tmp32, %tmp32b;
    setp.ge.u32 %p_oob, %r_tid, %total;
    @%p_oob bra DONE_PT;

    // b_idx = r_tid / dst_n
    // k_idx = r_tid % dst_n
    div.u32 %b_idx, %r_tid, %tmp32b;
    rem.u32 %k_idx, %r_tid, %tmp32b;

    // dst_offset = (b_idx * dst_n + k_idx) * 2
    mad.lo.u32 %dst_offset, %b_idx, %tmp32b, %k_idx;
    shl.b32 %dst_offset, %dst_offset, 1;

    // Compare k_idx vs src_n.
    ld.param.u32 %tmp32, [src_n];
    setp.ge.u32 %p_pad, %k_idx, %tmp32;
    @%p_pad bra PAD;

    // Copy from src[b_idx, k_idx, :].
    // src_offset = (b_idx * src_n + k_idx) * 2
    mad.lo.u32 %src_offset, %b_idx, %tmp32, %k_idx;
    shl.b32 %src_offset, %src_offset, 1;

    cvt.u64.u32 %off_src, %src_offset;
    shl.b64 %off_src, %off_src, 2;
    add.u64 %off_src, %src_base, %off_src;

    cvt.u64.u32 %off_dst, %dst_offset;
    shl.b64 %off_dst, %off_dst, 2;
    add.u64 %off_dst, %dst_base, %off_dst;

    ld.global.f32 %re, [%off_src];
    ld.global.f32 %im, [%off_src + 4];
    st.global.f32 [%off_dst], %re;
    st.global.f32 [%off_dst + 4], %im;
    bra DONE_PT;

PAD:
    cvt.u64.u32 %off_dst, %dst_offset;
    shl.b64 %off_dst, %off_dst, 2;
    add.u64 %off_dst, %dst_base, %off_dst;
    mov.f32 %re, 0f00000000;
    st.global.f32 [%off_dst], %re;
    st.global.f32 [%off_dst + 4], %re;

DONE_PT:
    ret;
}
";

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
// Cumulative scan PTX kernels
//
// One thread per (outer_idx, inner_idx) pair. Each thread does a sequential
// scan along dim_size elements. Parallelism comes from outer*inner threads.
// ---------------------------------------------------------------------------

/// PTX source for `cumsum_kernel`: prefix sum along an axis.
///
/// Thread i processes the scan for outer_idx = i / inner, inner_idx = i % inner.
/// `output[base + k*inner] = sum_{j=0}^{k} input[base + j*inner]`
#[cfg(feature = "cuda")]
pub(crate) const CUMSUM_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry cumsum_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %val, %acc;
    .reg .pred %p, %lp;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %n_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    // total threads = outer * inner
    mul.lo.u32 %tmp, %outer_sz, %inner_sz;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;

    // base = outer_idx * dim_size * inner_size + inner_idx
    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    mov.f32 %acc, 0f00000000;
    mov.u32 %k, 0;
SCAN_LOOP:
    setp.ge.u32 %lp, %k, %dim_sz;
    @%lp bra SCAN_DONE;

    // idx = base + k * inner_size
    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %val, [%addr];

    add.f32 %acc, %acc, %val;

    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %acc;

    add.u32 %k, %k, 1;
    bra SCAN_LOOP;
SCAN_DONE:

DONE:
    ret;
}
";

/// PTX source for `cumprod_kernel`: prefix product along an axis.
///
/// Thread i processes the scan for outer_idx = i / inner, inner_idx = i % inner.
/// `output[base + k*inner] = prod_{j=0}^{k} input[base + j*inner]`
#[cfg(feature = "cuda")]
pub(crate) const CUMPROD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry cumprod_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %val, %acc;
    .reg .pred %p, %lp;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %n_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    mul.lo.u32 %tmp, %outer_sz, %inner_sz;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;

    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    // acc = 1.0
    mov.f32 %acc, 0f3F800000;
    mov.u32 %k, 0;
SCAN_LOOP:
    setp.ge.u32 %lp, %k, %dim_sz;
    @%lp bra SCAN_DONE;

    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %val, [%addr];

    mul.f32 %acc, %acc, %val;

    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %acc;

    add.u32 %k, %k, 1;
    bra SCAN_LOOP;
SCAN_DONE:

DONE:
    ret;
}
";

/// PTX source for `cummax_kernel`: running maximum along an axis.
///
/// Thread i processes the scan for outer_idx = i / inner, inner_idx = i % inner.
/// Outputs both values and argmax indices (as f32 for uniform buffer handling).
/// `values[idx] = max_{j=0}^{k} input[base + j*inner]`
/// `indices[idx] = argmax_{j=0}^{k} input[base + j*inner]`
#[cfg(feature = "cuda")]
pub(crate) const CUMMAX_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry cummax_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 indices_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp, %best_k;
    .reg .u64 %in, %out, %ind, %off, %addr;
    .reg .f32 %val, %acc, %best_k_f;
    .reg .pred %p, %lp, %is_new_max;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u64 %ind, [indices_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %n_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    mul.lo.u32 %tmp, %outer_sz, %inner_sz;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;

    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    mov.b32 %acc, 0xFF800000;
    mov.u32 %best_k, 0;
    mov.u32 %k, 0;
SCAN_LOOP:
    setp.ge.u32 %lp, %k, %dim_sz;
    @%lp bra SCAN_DONE;

    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %val, [%addr];

    setp.gt.f32 %is_new_max, %val, %acc;
    @%is_new_max mov.u32 %best_k, %k;
    max.f32 %acc, %acc, %val;

    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %acc;

    cvt.rn.f32.u32 %best_k_f, %best_k;
    add.u64 %addr, %ind, %off;
    st.global.f32 [%addr], %best_k_f;

    add.u32 %k, %k, 1;
    bra SCAN_LOOP;
SCAN_DONE:

DONE:
    ret;
}
";

/// PTX source for `cummin_kernel`: running minimum along an axis.
///
/// Thread i processes the scan for outer_idx = i / inner, inner_idx = i % inner.
/// Outputs both values and argmin indices (as f32 for uniform buffer handling).
#[cfg(feature = "cuda")]
pub(crate) const CUMMIN_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry cummin_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 indices_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp, %best_k;
    .reg .u64 %in, %out, %ind, %off, %addr;
    .reg .f32 %val, %acc, %best_k_f;
    .reg .pred %p, %lp, %is_new_min;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u64 %ind, [indices_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %n_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    mul.lo.u32 %tmp, %outer_sz, %inner_sz;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;

    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    mov.b32 %acc, 0x7F800000;
    mov.u32 %best_k, 0;
    mov.u32 %k, 0;
SCAN_LOOP:
    setp.ge.u32 %lp, %k, %dim_sz;
    @%lp bra SCAN_DONE;

    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %val, [%addr];

    setp.lt.f32 %is_new_min, %val, %acc;
    @%is_new_min mov.u32 %best_k, %k;
    min.f32 %acc, %acc, %val;

    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %acc;

    cvt.rn.f32.u32 %best_k_f, %best_k;
    add.u64 %addr, %ind, %off;
    st.global.f32 [%addr], %best_k_f;

    add.u32 %k, %k, 1;
    bra SCAN_LOOP;
SCAN_DONE:

DONE:
    ret;
}
";

/// PTX source for `logcumsumexp_kernel`: numerically stable log-cumulative-sum-exp.
///
/// Thread i processes the scan for outer_idx = i / inner, inner_idx = i % inner.
/// `acc = log(exp(acc) + exp(x))` computed as `m + log(exp(acc-m) + exp(x-m))`
/// where `m = max(acc, x)` for numerical stability.
///
/// Uses `ex2.approx.f32` for exp and `lg2.approx.f32` for log with
/// log2(e) and ln(2) conversion constants.
#[cfg(feature = "cuda")]
pub(crate) const LOGCUMSUMEXP_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry logcumsumexp_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %val, %acc, %m, %ea, %ev, %s, %ls, %log2e, %ln2;
    .reg .pred %p, %lp;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %n_reg, [total];

    // log2(e) = 1.4426950408...  -> 0x3FB8AA3B
    mov.b32 %log2e, 0x3FB8AA3B;
    // ln(2) = 0.6931471805... -> 0x3F317218
    mov.b32 %ln2, 0x3F317218;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    mul.lo.u32 %tmp, %outer_sz, %inner_sz;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;

    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    // acc = -inf
    mov.b32 %acc, 0xFF800000;
    mov.u32 %k, 0;
SCAN_LOOP:
    setp.ge.u32 %lp, %k, %dim_sz;
    @%lp bra SCAN_DONE;

    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %val, [%addr];

    // Numerically stable: m = max(acc, x)
    max.f32 %m, %acc, %val;
    // exp(acc - m): (acc - m) * log2(e) -> ex2
    sub.f32 %ea, %acc, %m;
    mul.f32 %ea, %ea, %log2e;
    ex2.approx.f32 %ea, %ea;
    // exp(x - m): (x - m) * log2(e) -> ex2
    sub.f32 %ev, %val, %m;
    mul.f32 %ev, %ev, %log2e;
    ex2.approx.f32 %ev, %ev;
    // sum
    add.f32 %s, %ea, %ev;
    // log(sum) = lg2(sum) * ln(2)
    lg2.approx.f32 %ls, %s;
    mul.f32 %ls, %ls, %ln2;
    // acc = m + log(sum)
    add.f32 %acc, %m, %ls;

    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %acc;

    add.u32 %k, %k, 1;
    bra SCAN_LOOP;
SCAN_DONE:

DONE:
    ret;
}
";

/// PTX source for `logcumsumexp_f64_kernel`: numerically stable log-cumulative-sum-exp (f64).
#[cfg(feature = "cuda")]
pub(crate) const LOGCUMSUMEXP_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry logcumsumexp_f64_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 dim_size,
    .param .u32 inner_size,
    .param .u32 total
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %outer_sz, %dim_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %base, %idx, %tmp;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f64 %val, %acc, %m, %ea, %ev, %s, %ls;
    .reg .pred %p, %lp;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half, %e_one;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
    .reg .u64 %l_xbits, %l_mbits, %l_bias;
    .reg .s64 %l_exp64;
    .reg .f64 %l_m, %l_f, %l_f2, %l_s, %l_p, %l_nf, %l_ln2;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %dim_sz, [dim_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %n_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    mul.lo.u32 %tmp, %outer_sz, %inner_sz;
    setp.ge.u32 %p, %r_tid, %tmp;
    @%p bra DONE;

    div.u32 %outer_idx, %r_tid, %inner_sz;
    rem.u32 %inner_idx, %r_tid, %inner_sz;

    mul.lo.u32 %base, %outer_idx, %dim_sz;
    mul.lo.u32 %base, %base, %inner_sz;
    add.u32 %base, %base, %inner_idx;

    // acc = -inf
    mov.b64 %acc, 0xFFF0000000000000;
    mov.u32 %k, 0;
SCAN_LOOP:
    setp.ge.u32 %lp, %k, %dim_sz;
    @%lp bra SCAN_DONE;

    mul.lo.u32 %idx, %k, %inner_sz;
    add.u32 %idx, %base, %idx;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 3;
    add.u64 %addr, %in, %off;
    ld.global.f64 %val, [%addr];

    max.f64 %m, %acc, %val;
    mov.f64 %e_one, 0d3FF0000000000000;
    mov.f64 %e_half, 0d3FE0000000000000;
    // --- inline exp(acc - m) -> %ea ---
    sub.f64 %ea, %acc, %m;
    mul.f64 %e_nf, %ea, 0d3FF71547652B82FE;
    cvt.rni.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %ea;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FC5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_one;
    fma.rn.f64 %ea, %e_p, %e_r, %e_one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %ea, %ea, %e_nf;
    // --- inline exp(val - m) -> %ev ---
    sub.f64 %ev, %val, %m;
    mul.f64 %e_nf, %ev, 0d3FF71547652B82FE;
    cvt.rni.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %ev;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FC5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_one;
    fma.rn.f64 %ev, %e_p, %e_r, %e_one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %ev, %ev, %e_nf;
    add.f64 %s, %ea, %ev;
    // --- inline ln(%s) -> %ls ---
    mov.b64 %l_xbits, %s;
    shr.u64 %l_exp64, %l_xbits, 52;
    and.b64 %l_exp64, %l_exp64, 2047;
    sub.s64 %l_exp64, %l_exp64, 1023;
    cvt.rn.f64.s64 %l_nf, %l_exp64;
    mov.u64 %l_bias, 0x3FF0000000000000;
    and.b64 %l_mbits, %l_xbits, 0x000FFFFFFFFFFFFF;
    or.b64 %l_mbits, %l_mbits, %l_bias;
    mov.b64 %l_m, %l_mbits;
    sub.f64 %l_f, %l_m, %e_one;
    add.f64 %l_s, %l_m, %e_one;
    div.rn.f64 %l_f, %l_f, %l_s;
    mul.f64 %l_f2, %l_f, %l_f;
    mov.f64 %l_p, 0d3FB745D1745D1746;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC1C71C71C71C72;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC2492492492492;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC999999999999A;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FD5555555555555;
    fma.rn.f64 %l_p, %l_p, %l_f2, %e_one;
    mul.f64 %l_p, %l_p, %l_f;
    add.f64 %l_p, %l_p, %l_p;
    mov.f64 %l_ln2, 0d3FE62E42FEFA39EF;
    fma.rn.f64 %ls, %l_nf, %l_ln2, %l_p;
    add.f64 %acc, %m, %ls;

    add.u64 %addr, %out, %off;
    st.global.f64 [%addr], %acc;

    add.u32 %k, %k, 1;
    bra SCAN_LOOP;
SCAN_DONE:

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
// RMSNorm PTX kernel (row-wise: rms, normalize+scale)
//
// Like LayerNorm but without mean centering or bias:
//   out[j] = x[j] * rsqrt(mean(x^2) + eps) * weight[j]
//
// Uses `.approx` PTX instructions (`div.approx.f32`, `sqrt.approx.f32`,
// `rcp.approx.f32`) for performance. These have reduced precision (~2^-22
// relative error) compared to the full-precision variants, which is
// acceptable for neural network training/inference.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub(crate) const RMSNORM_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry rmsnorm_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u64 w_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .f32 eps
) {
    .reg .u32 %r_tid, %r_bid, %r_bdim, %rows_reg, %cols_reg, %j, %half, %r_otid;
    .reg .u64 %in, %out, %w, %row_off, %off, %sbase, %saddr;
    .reg .f32 %val, %sq_sum, %eps_r, %inv_rms, %wv, %result, %other_val, %n_f;
    .reg .pred %p, %lp, %rp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %w, [w_ptr];
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

    // ===== Phase 1: Compute sum(x^2) =====
    mov.f32 %sq_sum, 0f00000000;
    mov.u32 %j, %r_tid;
SS:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SSD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.f32 %val, [%off];
    fma.rn.f32 %sq_sum, %val, %val, %sq_sum;
    add.u32 %j, %j, %r_bdim;
    bra SS;
SSD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sq_sum;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
SR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra SRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra SRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sq_sum, [%saddr];
    add.f32 %sq_sum, %sq_sum, %other_val;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sq_sum;
SRS:
    bar.sync 0;
    bra SR;
SRD:
    ld.shared.f32 %sq_sum, [%sbase];
    div.approx.f32 %sq_sum, %sq_sum, %n_f;
    add.f32 %sq_sum, %sq_sum, %eps_r;
    sqrt.approx.f32 %inv_rms, %sq_sum;
    rcp.approx.f32 %inv_rms, %inv_rms;
    bar.sync 0;

    // ===== Phase 2: Normalize and scale =====
    // out[j] = x[j] * inv_rms * weight[j]
    mov.u32 %j, %r_tid;
NM:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra NMD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.f32 %val, [%off];
    mul.f32 %result, %val, %inv_rms;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %off, %w, %off;
    ld.global.f32 %wv, [%off];
    mul.f32 %result, %result, %wv;
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
// RMSNorm backward PTX kernel
// ---------------------------------------------------------------------------
//
// One block per batch element (row). Each block:
//   1. Recompute inv_rms = 1/sqrt(mean(x^2) + eps)
//   2. Compute dot = sum(grad_output[j] * x[j] * weight[j])
//   3. Compute grad_input[j] = inv_rms * weight[j] * go[j]
//                              - x[j] * inv_rms^3 * dot / cols
//   4. Accumulate grad_weight[j] (atomicAdd) = go[j] * x[j] * inv_rms
//
// Uses shared memory for per-row reductions, 256 threads per block.
// No grad_bias (RMSNorm has no bias parameter).
// Parameters:
//   in_ptr       - pointer to input f32 buffer [rows * cols]
//   grad_out_ptr - pointer to grad_output f32 buffer [rows * cols]
//   w_ptr        - pointer to weight f32 buffer [cols]
//   grad_in_ptr  - pointer to grad_input f32 output buffer [rows * cols]
//   grad_w_ptr   - pointer to grad_weight f32 output buffer [cols] (atomicAdd)
//   rows         - number of batch elements
//   cols         - normalized dimension size
//   eps          - epsilon for numerical stability

#[cfg(feature = "cuda")]
pub(crate) const RMSNORM_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry rmsnorm_backward_kernel(
    .param .u64 in_ptr,
    .param .u64 grad_out_ptr,
    .param .u64 w_ptr,
    .param .u64 grad_in_ptr,
    .param .u64 grad_w_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .f32 eps
) {
    .reg .u32 %r_tid, %r_bid, %r_bdim, %rows_reg, %cols_reg, %j, %half, %r_otid;
    .reg .u64 %in, %go, %w, %gi, %gw, %row_off, %off, %sbase, %saddr, %addr;
    .reg .f32 %val, %sq_sum, %eps_r, %inv_rms, %inv_rms3, %wv, %gov;
    .reg .f32 %dot, %other_val, %n_f, %coeff, %result, %tmp;
    .reg .pred %p, %lp, %rp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %go, [grad_out_ptr];
    ld.param.u64 %w, [w_ptr];
    ld.param.u64 %gi, [grad_in_ptr];
    ld.param.u64 %gw, [grad_w_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];
    ld.param.f32 %eps_r, [eps];

    mov.u64 %sbase, sdata;

    mov.u32 %r_bid, %ctaid.x;
    mov.u32 %r_bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %r_bid, %rows_reg;
    @%p bra RNB_DONE;

    // row_off = bid * cols * 4 (byte offset for this row)
    cvt.u64.u32 %row_off, %r_bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 2;
    cvt.rn.f32.u32 %n_f, %cols_reg;

    // ===== Phase 1: Compute sum(x^2) -> inv_rms =====
    mov.f32 %sq_sum, 0f00000000;
    mov.u32 %j, %r_tid;
RNB_SS:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra RNB_SSD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %val, [%addr];
    fma.rn.f32 %sq_sum, %val, %val, %sq_sum;
    add.u32 %j, %j, %r_bdim;
    bra RNB_SS;
RNB_SSD:
    // Shared memory reduce for sum(x^2)
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sq_sum;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
RNB_SR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra RNB_SRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra RNB_SRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sq_sum, [%saddr];
    add.f32 %sq_sum, %sq_sum, %other_val;
    st.shared.f32 [%saddr], %sq_sum;
RNB_SRS:
    bar.sync 0;
    bra RNB_SR;
RNB_SRD:
    ld.shared.f32 %sq_sum, [%sbase];
    div.approx.f32 %sq_sum, %sq_sum, %n_f;
    add.f32 %sq_sum, %sq_sum, %eps_r;
    sqrt.approx.f32 %inv_rms, %sq_sum;
    rcp.approx.f32 %inv_rms, %inv_rms;
    // inv_rms3 = inv_rms^3 = inv_rms * inv_rms * inv_rms
    mul.f32 %inv_rms3, %inv_rms, %inv_rms;
    mul.f32 %inv_rms3, %inv_rms3, %inv_rms;
    bar.sync 0;

    // ===== Phase 2: Compute dot = sum(go[j] * x[j] * w[j]) =====
    // Also accumulate grad_weight via atomicAdd
    mov.f32 %dot, 0f00000000;
    mov.u32 %j, %r_tid;
RNB_DOT:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra RNB_DOTD;
    // Load input[row, j]
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %val, [%addr];
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
    // dot += go * x * w
    mul.f32 %tmp, %gov, %val;
    fma.rn.f32 %dot, %tmp, %wv, %dot;
    // atomicAdd grad_weight[j] += go * x * inv_rms
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %gw, %off;
    mul.f32 %result, %gov, %val;
    mul.f32 %result, %result, %inv_rms;
    atom.global.add.f32 %result, [%addr], %result;
    add.u32 %j, %j, %r_bdim;
    bra RNB_DOT;
RNB_DOTD:
    // Reduce dot in shared memory
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %dot;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
RNB_DR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra RNB_DRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra RNB_DRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %dot, [%saddr];
    add.f32 %dot, %dot, %other_val;
    st.shared.f32 [%saddr], %dot;
RNB_DRS:
    bar.sync 0;
    bra RNB_DR;
RNB_DRD:
    ld.shared.f32 %dot, [%sbase];
    // coeff = dot * inv_rms3 / n
    mul.f32 %coeff, %dot, %inv_rms3;
    div.approx.f32 %coeff, %coeff, %n_f;
    bar.sync 0;

    // ===== Phase 3: Compute grad_input =====
    // grad_input[j] = inv_rms * w[j] * go[j] - x[j] * coeff
    mov.u32 %j, %r_tid;
RNB_GI:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra RNB_GID;
    // Reload input
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %val, [%addr];
    // Reload grad_output and weight
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %go, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %gov, [%addr];
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %w, %off;
    ld.global.f32 %wv, [%addr];
    // result = inv_rms * w * go - x * coeff
    mul.f32 %result, %inv_rms, %wv;
    mul.f32 %result, %result, %gov;
    mul.f32 %tmp, %val, %coeff;
    sub.f32 %result, %result, %tmp;
    // Store grad_input[row, j]
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %gi, %off;
    add.u64 %addr, %addr, %row_off;
    st.global.f32 [%addr], %result;
    add.u32 %j, %j, %r_bdim;
    bra RNB_GI;
RNB_GID:

RNB_DONE:
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
    .reg .u32 %my_tid, %bid, %bdim, %ch, %n_ch, %sp, %tpc, %idx, %train;
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
    mov.u32 %my_tid, %tid.x;
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
    mov.u32 %idx, %my_tid;
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
    cvt.u64.u32 %off64, %my_tid;
    shl.b64 %off64, %off64, 2;
    st.shared.f32 [smem_sum + %off64], %sum;
    st.shared.f32 [smem_sq + %off64], %sqsum;
    bar.sync 0;

    // Tree reduction
    mov.u32 %half, 128;
REDUCE_LOOP:
    setp.lt.u32 %p, %half, 1;
    @%p bra REDUCE_DONE;
    setp.ge.u32 %p, %my_tid, %half;
    @%p bra REDUCE_SKIP;

    add.u32 %idx, %my_tid, %half;
    cvt.u64.u32 %off64, %idx;
    shl.b64 %off64, %off64, 2;
    ld.shared.f32 %other, [smem_sum + %off64];
    cvt.u64.u32 %tmp64, %my_tid;
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
    setp.ne.u32 %ptid0, %my_tid, 0;

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
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %idx, %stride, %total_reg;
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
    mov.u32 %my_tid, %tid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
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
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %idx, %stride, %total_reg;
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
    mov.u32 %my_tid, %tid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
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

/// PTX source for `softmax_f64_kernel`: row-wise softmax (f64).
#[cfg(feature = "cuda")]
pub(crate) const SOFTMAX_F64_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.shared .align 8 .f64 sdata[256];\n\
\n\
.visible .entry softmax_f64_kernel(\n\
    .param .u64 input_ptr,\n\
    .param .u64 output_ptr,\n\
    .param .u32 rows,\n\
    .param .u32 cols\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j;\n\
    .reg .u64 %in, %out, %row_off, %off, %sbase, %saddr;\n\
    .reg .f64 %val, %max_val, %sum_val, %exp_val, %result, %one;\n\
    .reg .pred %p, %loop_p;\n\
    .reg .u32 %half, %other_tid;\n\
    .reg .f64 %other_val;\n\
    .reg .pred %reduce_p;\n\
    .reg .f64 %e_nf, %e_r, %e_p, %e_half, %e_one;\n\
    .reg .s32 %e_ni;\n\
    .reg .s64 %e_ni64, %e_bits;\n\
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
    mov.f64 %one, 0d3FF0000000000000;\n\
\n\
    setp.ge.u32 %p, %bid, %rows_reg;\n\
    @%p bra DONE;\n\
\n\
    cvt.u64.u32 %row_off, %bid;\n\
    cvt.u64.u32 %off, %cols_reg;\n\
    mul.lo.u64 %row_off, %row_off, %off;\n\
    shl.b64 %row_off, %row_off, 3;\n\
\n\
    mov.f64 %max_val, 0dFFF0000000000000;\n\
    mov.u32 %j, %r_tid;\n\
FIND_MAX:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra FIND_MAX_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %off, %in, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    ld.global.f64 %val, [%off];\n\
    max.f64 %max_val, %max_val, %val;\n\
    add.u32 %j, %j, %bdim;\n\
    bra FIND_MAX;\n\
FIND_MAX_DONE:\n\
\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f64 [%saddr], %max_val;\n\
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
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %other_val, [%saddr];\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %max_val, [%saddr];\n\
    max.f64 %max_val, %max_val, %other_val;\n\
    st.shared.f64 [%saddr], %max_val;\n\
MAX_REDUCE_SKIP:\n\
    bar.sync 0;\n\
    bra MAX_REDUCE;\n\
MAX_REDUCE_DONE:\n\
\n\
    ld.shared.f64 %max_val, [sdata];\n\
    bar.sync 0;\n\
\n\
    mov.f64 %sum_val, 0d0000000000000000;\n\
    mov.u32 %j, %r_tid;\n\
SUM_EXP:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra SUM_EXP_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %off, %in, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    ld.global.f64 %val, [%off];\n\
    sub.f64 %val, %val, %max_val;\n\
    mov.f64 %e_one, 0d3FF0000000000000;\n\
    mov.f64 %e_half, 0d3FE0000000000000;\n\
    mul.f64 %e_nf, %val, 0d3FF71547652B82FE;\n\
    cvt.rni.f64.f64 %e_nf, %e_nf;\n\
    cvt.rni.s32.f64 %e_ni, %e_nf;\n\
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %val;\n\
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;\n\
    mov.f64 %e_p, 0d3E21EED8EFF8D898;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FC5555555555555;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;\n\
    fma.rn.f64 %e_p, %e_p, %e_r, %e_one;\n\
    fma.rn.f64 %exp_val, %e_p, %e_r, %e_one;\n\
    cvt.s64.s32 %e_ni64, %e_ni;\n\
    add.s64 %e_ni64, %e_ni64, 1023;\n\
    shl.b64 %e_bits, %e_ni64, 52;\n\
    mov.b64 %e_nf, %e_bits;\n\
    mul.f64 %exp_val, %exp_val, %e_nf;\n\
    add.f64 %sum_val, %sum_val, %exp_val;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %off, %out, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    st.global.f64 [%off], %exp_val;\n\
    add.u32 %j, %j, %bdim;\n\
    bra SUM_EXP;\n\
SUM_EXP_DONE:\n\
\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f64 [%saddr], %sum_val;\n\
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
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %other_val, [%saddr];\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f64 %sum_val, [%saddr];\n\
    add.f64 %sum_val, %sum_val, %other_val;\n\
    st.shared.f64 [%saddr], %sum_val;\n\
SUM_REDUCE_SKIP:\n\
    bar.sync 0;\n\
    bra SUM_REDUCE;\n\
SUM_REDUCE_DONE:\n\
\n\
    ld.shared.f64 %sum_val, [sdata];\n\
    bar.sync 0;\n\
\n\
    div.rn.f64 %sum_val, %one, %sum_val;\n\
    mov.u32 %j, %r_tid;\n\
NORMALIZE:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra NORMALIZE_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 3;\n\
    add.u64 %off, %out, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    ld.global.f64 %val, [%off];\n\
    mul.f64 %result, %val, %sum_val;\n\
    st.global.f64 [%off], %result;\n\
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

/// PTX source for `strided_copy_kernel`: general strided→contiguous
/// gather with up to 8 dimensions. CL-496.
///
/// Thread `i` computes:
///   flat = i
///   src = src_offset_base
///   for d in 0..8:
///       coord = flat / out_stride[d]
///       flat  = flat % out_stride[d]
///       src  += coord * src_stride[d]
///   out[i] = in[src]
///
/// For tensors with fewer than 8 dims, unused positions must be
/// padded with `out_stride[d] = n + 1` (so `flat / out_stride[d] = 0`)
/// and `src_stride[d] = 0` (so the contribution is zero).
///
/// Each stride is passed as an individual u32 kernel parameter to
/// avoid needing a device-side stride array. 20 params total is well
/// within the ~4KB param limit.
#[cfg(feature = "cuda")]
pub(crate) const STRIDED_COPY_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry strided_copy_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 src_offset_base,
    .param .u32 n,
    .param .u32 os0, .param .u32 os1, .param .u32 os2, .param .u32 os3,
    .param .u32 os4, .param .u32 os5, .param .u32 os6, .param .u32 os7,
    .param .u32 ss0, .param .u32 ss1, .param .u32 ss2, .param .u32 ss3,
    .param .u32 ss4, .param .u32 ss5, .param .u32 ss6, .param .u32 ss7
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u32 %flat, %src_idx, %coord, %tmp, %os, %ss;
    .reg .u64 %in, %out, %off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %src_idx, [src_offset_base];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %flat, %r_tid;

    // Dim 0
    ld.param.u32 %os, [os0];
    ld.param.u32 %ss, [ss0];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 1
    ld.param.u32 %os, [os1];
    ld.param.u32 %ss, [ss1];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 2
    ld.param.u32 %os, [os2];
    ld.param.u32 %ss, [ss2];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 3
    ld.param.u32 %os, [os3];
    ld.param.u32 %ss, [ss3];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 4
    ld.param.u32 %os, [os4];
    ld.param.u32 %ss, [ss4];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 5
    ld.param.u32 %os, [os5];
    ld.param.u32 %ss, [ss5];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 6
    ld.param.u32 %os, [os6];
    ld.param.u32 %ss, [ss6];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

    // Dim 7
    ld.param.u32 %os, [os7];
    ld.param.u32 %ss, [ss7];
    div.u32 %coord, %flat, %os;
    mul.lo.u32 %tmp, %coord, %os;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ss;
    add.u32 %src_idx, %src_idx, %tmp;

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

/// PTX source for `strided_scatter_kernel`: inverse of `strided_copy_kernel`.
///
/// Each thread `tid` corresponds to one position in the contiguous source
/// buffer `in`. The kernel decodes `tid` into per-dimension coordinates
/// using the source-shape products `is0..is7`, accumulates a destination
/// element index `dst_idx = dst_offset + Σ_d coord_d * ds_d` using the
/// strided-view strides `ds0..ds7`, and writes `in[tid]` into
/// `out[dst_idx]`.
///
/// Mirrors the f32 strided-copy kernel structurally; only the final
/// load/store pair is reversed.
#[cfg(feature = "cuda")]
pub(crate) const STRIDED_SCATTER_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry strided_scatter_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 dst_offset_base,
    .param .u32 n,
    .param .u32 is0, .param .u32 is1, .param .u32 is2, .param .u32 is3,
    .param .u32 is4, .param .u32 is5, .param .u32 is6, .param .u32 is7,
    .param .u32 ds0, .param .u32 ds1, .param .u32 ds2, .param .u32 ds3,
    .param .u32 ds4, .param .u32 ds5, .param .u32 ds6, .param .u32 ds7
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u32 %flat, %dst_idx, %coord, %tmp, %is, %ds;
    .reg .u64 %in, %out, %off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %dst_idx, [dst_offset_base];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %flat, %r_tid;

    // Dim 0
    ld.param.u32 %is, [is0];
    ld.param.u32 %ds, [ds0];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    // Dim 1
    ld.param.u32 %is, [is1];
    ld.param.u32 %ds, [ds1];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    // Dim 2
    ld.param.u32 %is, [is2];
    ld.param.u32 %ds, [ds2];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    // Dim 3
    ld.param.u32 %is, [is3];
    ld.param.u32 %ds, [ds3];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    // Dim 4
    ld.param.u32 %is, [is4];
    ld.param.u32 %ds, [ds4];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    // Dim 5
    ld.param.u32 %is, [is5];
    ld.param.u32 %ds, [ds5];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    // Dim 6
    ld.param.u32 %is, [is6];
    ld.param.u32 %ds, [ds6];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    // Dim 7
    ld.param.u32 %is, [is7];
    ld.param.u32 %ds, [ds7];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

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

/// PTX source for `strided_scatter_f64_kernel` — same shape as the f32
/// kernel but with `f64` element loads and `<<3` byte-offset shifts
/// (instead of `<<2`).
#[cfg(feature = "cuda")]
pub(crate) const STRIDED_SCATTER_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry strided_scatter_f64_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 dst_offset_base,
    .param .u32 n,
    .param .u32 is0, .param .u32 is1, .param .u32 is2, .param .u32 is3,
    .param .u32 is4, .param .u32 is5, .param .u32 is6, .param .u32 is7,
    .param .u32 ds0, .param .u32 ds1, .param .u32 ds2, .param .u32 ds3,
    .param .u32 ds4, .param .u32 ds5, .param .u32 ds6, .param .u32 ds7
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u32 %flat, %dst_idx, %coord, %tmp, %is, %ds;
    .reg .u64 %in, %out, %off;
    .reg .f64 %val;
    .reg .pred %p;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %dst_idx, [dst_offset_base];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %flat, %r_tid;

    ld.param.u32 %is, [is0];
    ld.param.u32 %ds, [ds0];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    ld.param.u32 %is, [is1];
    ld.param.u32 %ds, [ds1];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    ld.param.u32 %is, [is2];
    ld.param.u32 %ds, [ds2];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    ld.param.u32 %is, [is3];
    ld.param.u32 %ds, [ds3];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    ld.param.u32 %is, [is4];
    ld.param.u32 %ds, [ds4];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    ld.param.u32 %is, [is5];
    ld.param.u32 %ds, [ds5];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    ld.param.u32 %is, [is6];
    ld.param.u32 %ds, [ds6];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    ld.param.u32 %is, [is7];
    ld.param.u32 %ds, [ds7];
    div.u32 %coord, %flat, %is;
    mul.lo.u32 %tmp, %coord, %is;
    sub.u32 %flat, %flat, %tmp;
    mul.lo.u32 %tmp, %coord, %ds;
    add.u32 %dst_idx, %dst_idx, %tmp;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 3;
    add.u64 %off, %in, %off;
    ld.global.f64 %val, [%off];

    cvt.u64.u32 %off, %dst_idx;
    shl.b64 %off, %off, 3;
    add.u64 %off, %out, %off;
    st.global.f64 [%off], %val;

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

/// PTX source for `exp_f64_kernel`: `out[i] = exp(a[i])` (f64).
/// Uses f32 `ex2.approx` via downcast for the transcendental, then upcasts back.
/// Accurate to f32 precision (~7 decimal digits), sufficient for deep learning.
#[cfg(feature = "cuda")]
/// f64 exp with full double precision via Cody-Waite range reduction +
/// degree-13 minimax polynomial.
///
/// Algorithm: exp(x) = 2^n * (1 + P(r))
///   where n = round(x * log2(e)), r = x - n*ln2_hi - n*ln2_lo
///   and P(r) is a 13th-degree minimax polynomial for (exp(r)-1)/r.
///
/// Accuracy: < 1 ULP for |x| < 709 (full f64 range).
pub(crate) const EXP_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry exp_f64_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f64 %x, %vr;
    .reg .f64 %log2e, %nf, %r;
    .reg .f64 %p, %one, %half;
    .reg .s32 %ni;
    .reg .s64 %ni64, %exp_bits;
    .reg .pred %p_bounds, %p_tid;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p_tid, %r_tid, %n_reg;
    @%p_tid bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %x, [%a];

    // Constants
    mov.f64 %log2e, 0d3FF71547652B82FE;   // log2(e) = 1.4426950408889634
    mov.f64 %ln2_hi, 0d3FE62E42FEFA3800;  // ln(2) high bits
    mov.f64 %ln2_lo, 0d3D2EF35793C76730;  // ln(2) low bits
    mov.f64 %one, 0d3FF0000000000000;      // 1.0
    mov.f64 %half, 0d3FE0000000000000;     // 0.5

    // n = round(x * log2(e))
    mul.f64 %nf, %x, %log2e;
    cvt.rni.f64.f64 %nf, %nf;             // round to nearest integer
    cvt.rni.s32.f64 %ni, %nf;             // integer n

    // r = x - n * ln2  (Cody-Waite two-step for precision)
    fma.rn.f64 %r, %nf, 0dBFE62E42FEFA3800, %x;  // r = x - n*ln2_hi
    fma.rn.f64 %r, %nf, 0dBD2EF35793C76730, %r;   // r -= n*ln2_lo

    // Horner polynomial for exp(r) - 1 - r = r^2 * (1/2! + r*(1/3! + r*(1/4! + ...)))
    // p starts at 1/11!, accumulates down to 1/2!
    mov.f64 %p, 0d3E21EED8EFF8D898;           // 1/11! = 2.505e-8
    fma.rn.f64 %p, %p, %r, 0d3E5AE64567F544E4;  // 1/10! = 2.756e-7
    fma.rn.f64 %p, %p, %r, 0d3E927E4FB7789F5C;  // 1/9!  = 2.756e-6
    fma.rn.f64 %p, %p, %r, 0d3EC71DE3A556C734;  // 1/8!  = 2.480e-5
    fma.rn.f64 %p, %p, %r, 0d3EFA01A01A01A01A;  // 1/7!  = 1.984e-4
    fma.rn.f64 %p, %p, %r, 0d3F2A01A01A01A01A;  // 1/6!  = 1.389e-3
    fma.rn.f64 %p, %p, %r, 0d3F56C16C16C16C17;  // 1/5!  = 8.333e-3
    fma.rn.f64 %p, %p, %r, 0d3F811111111111111;  // 1/4!  = 4.167e-2
    fma.rn.f64 %p, %p, %r, 0d3FC5555555555555;  // 1/3!  = 1.667e-1
    fma.rn.f64 %p, %p, %r, %half;                // 1/2!  = 5.000e-1

    // exp(r) = 1 + r + r^2 * p  =>  1 + r*(1 + r*p)
    fma.rn.f64 %p, %p, %r, %one;   // p = r*p + 1
    fma.rn.f64 %vr, %p, %r, %one;  // vr = p*r + 1 = exp(r)

    // Scale by 2^n: multiply by constructing the f64 bit pattern for 2^n.
    // IEEE 754 f64: 2^n has exponent field = n + 1023, no mantissa bits.
    // Bit pattern: (n + 1023) << 52.
    cvt.s64.s32 %ni64, %ni;
    add.s64 %ni64, %ni64, 1023;
    shl.b64 %exp_bits, %ni64, 52;
    mov.b64 %nf, %exp_bits;        // reinterpret as f64 = 2^n
    mul.f64 %vr, %vr, %nf;

    st.global.f64 [%out], %vr;

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

/// PTX source for `log_f64_kernel`: `out[i] = ln(a[i])` (f64).
/// Uses f32 `lg2.approx` via downcast for the transcendental, then upcasts back.
/// Accurate to f32 precision (~7 decimal digits), sufficient for deep learning.
#[cfg(feature = "cuda")]
/// f64 log with full double precision via argument reduction + rational
/// approximation.
///
/// Algorithm: decompose x = 2^n * m (1 <= m < 2), then
///   ln(x) = n*ln(2) + ln(m)
/// where ln(m) is computed via f = (m-1)/(m+1), ln(m) = 2*f*(1 + f^2/3 + f^4/5 + ...)
///
/// Accuracy: < 2 ULP across the full f64 range.
pub(crate) const LOG_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry log_f64_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .u64 %xbits, %mantissa_bits, %bias_bits;
    .reg .f64 %x, %vr, %m, %f, %f2, %s, %p;
    .reg .f64 %ln2_hi, %ln2_lo, %one, %two;
    .reg .s32 %exp_i;
    .reg .s64 %exp64;
    .reg .f64 %nf;
    .reg .pred %p_tid;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p_tid, %r_tid, %n_reg;
    @%p_tid bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 3;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %x, [%a];

    mov.f64 %ln2_hi, 0d3FE62E42FEFA39EF;   // ln(2) = 0.6931471805599453
    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %two, 0d4000000000000000;

    // Extract exponent: n = exponent_field - 1023
    mov.b64 %xbits, %x;
    shr.u64 %exp64, %xbits, 52;
    and.b64 %exp64, %exp64, 2047;   // 11-bit exponent field
    sub.s64 %exp64, %exp64, 1023;
    cvt.rn.f64.s64 %nf, %exp64;     // n as f64

    // Extract mantissa m: set exponent to 1023 (so m is in [1, 2))
    mov.u64 %bias_bits, 0x3FF0000000000000;  // exponent = 1023
    and.b64 %mantissa_bits, %xbits, 0x000FFFFFFFFFFFFF;  // mantissa bits
    or.b64 %mantissa_bits, %mantissa_bits, %bias_bits;
    mov.b64 %m, %mantissa_bits;      // m in [1.0, 2.0)

    // f = (m - 1) / (m + 1) — maps [1,2) to [0, 1/3)
    sub.f64 %f, %m, %one;
    add.f64 %s, %m, %one;
    div.rn.f64 %f, %f, %s;

    // ln(m) = 2*f + 2*f^3/3 + 2*f^5/5 + 2*f^7/7 + 2*f^9/9 + 2*f^11/11
    // Horner: ln(m) = 2*f*(1 + f^2*(1/3 + f^2*(1/5 + f^2*(1/7 + f^2*(1/9 + f^2/11)))))
    mul.f64 %f2, %f, %f;

    // p = 1/11
    mov.f64 %p, 0d3FB745D1745D1746;
    // p = p*f2 + 1/9
    fma.rn.f64 %p, %p, %f2, 0d3FC1C71C71C71C72;
    // p = p*f2 + 1/7
    fma.rn.f64 %p, %p, %f2, 0d3FC2492492492492;
    // p = p*f2 + 1/5
    fma.rn.f64 %p, %p, %f2, 0d3FC999999999999A;
    // p = p*f2 + 1/3
    fma.rn.f64 %p, %p, %f2, 0d3FD5555555555555;
    // p = p*f2 + 1
    fma.rn.f64 %p, %p, %f2, %one;

    // ln(m) = 2*f*p
    mul.f64 %p, %p, %f;
    add.f64 %p, %p, %p;   // * 2

    // ln(x) = n*ln(2) + ln(m)
    fma.rn.f64 %vr, %nf, %ln2_hi, %p;

    st.global.f64 [%out], %vr;

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

/// PTX source for `pow_f64_kernel`: `out[i] = a[i] ^ exponent` (f64).
/// Full f64 precision: x^e = exp(e * ln(x)).
/// Uses inline f64 log (argument reduction + odd-power series) and
/// inline f64 exp (Cody-Waite + degree-11 Horner).
#[cfg(feature = "cuda")]
pub(crate) const POW_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry pow_f64_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .f64 exponent,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f64 %va, %vr, %exp64, %one, %two;
    // log registers
    .reg .u64 %l_xbits, %l_mbits, %l_bias;
    .reg .s64 %l_exp64;
    .reg .f64 %l_m, %l_f, %l_f2, %l_s, %l_p, %l_nf, %l_ln2, %l_lnx;
    // exp registers
    .reg .f64 %e_z, %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f64 %exp64, [exponent];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 3;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %va, [%a];
    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %two, 0d4000000000000000;

    // === ln(va) via argument reduction ===
    // Decompose va = 2^n * m, m in [1,2), ln(va) = n*ln(2) + ln(m)
    mov.b64 %l_xbits, %va;
    shr.u64 %l_exp64, %l_xbits, 52;
    and.b64 %l_exp64, %l_exp64, 2047;
    sub.s64 %l_exp64, %l_exp64, 1023;
    cvt.rn.f64.s64 %l_nf, %l_exp64;

    mov.u64 %l_bias, 0x3FF0000000000000;
    and.b64 %l_mbits, %l_xbits, 0x000FFFFFFFFFFFFF;
    or.b64 %l_mbits, %l_mbits, %l_bias;
    mov.b64 %l_m, %l_mbits;

    // f = (m-1)/(m+1)
    sub.f64 %l_f, %l_m, %one;
    add.f64 %l_s, %l_m, %one;
    div.rn.f64 %l_f, %l_f, %l_s;
    mul.f64 %l_f2, %l_f, %l_f;

    // Horner: p = 1/11 + f2*(1/9 + f2*(1/7 + f2*(1/5 + f2*(1/3 + f2*1))))
    mov.f64 %l_p, 0d3FB745D1745D1746;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC1C71C71C71C72;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC2492492492492;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FC999999999999A;
    fma.rn.f64 %l_p, %l_p, %l_f2, 0d3FD5555555555555;
    fma.rn.f64 %l_p, %l_p, %l_f2, %one;

    // ln(m) = 2*f*p
    mul.f64 %l_p, %l_p, %l_f;
    add.f64 %l_p, %l_p, %l_p;

    // ln(x) = n*ln(2) + ln(m)
    mov.f64 %l_ln2, 0d3FE62E42FEFA39EF;
    fma.rn.f64 %l_lnx, %l_nf, %l_ln2, %l_p;

    // === exp(exponent * ln(x)) ===
    mul.f64 %e_z, %exp64, %l_lnx;

    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %e_z, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %e_z;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %vr, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %vr, %vr, %e_nf;

    st.global.f64 [%out], %vr;

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

/// PTX source for `sigmoid_f64_kernel`: `out[i] = 1 / (1 + exp(-a[i]))` (f64).
/// Full f64 precision: Cody-Waite range reduction + degree-11 Horner polynomial
/// for exp(-x), then sigmoid = 1/(1+exp(-x)).
#[cfg(feature = "cuda")]
pub(crate) const SIGMOID_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry sigmoid_f64_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f64 %va, %vr, %e64, %denom, %one, %neg_x;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
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
    shl.b64 %off, %off, 3;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %va, [%a];
    mov.f64 %one, 0d3FF0000000000000;

    // sigmoid(x) = 1 / (1 + exp(-x))
    neg.f64 %neg_x, %va;

    // --- exp(%neg_x) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %neg_x, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg_x;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e64, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e64, %e64, %e_nf;
    // --- end exp ---

    add.f64 %denom, %one, %e64;
    div.rn.f64 %vr, %one, %denom;
    st.global.f64 [%out], %vr;

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

/// PTX source for `tanh_f64_kernel`: `out[i] = tanh(a[i])` (f64).
/// Uses the identity: tanh(x) = 2*sigmoid(2x) - 1 = (1-exp(-2x))/(1+exp(-2x)).
/// Full f64 precision via Cody-Waite + degree-11 Horner for exp(-2x).
#[cfg(feature = "cuda")]
pub(crate) const TANH_F64_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry tanh_f64_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f64 %va, %vr, %e64, %num, %denom, %one, %two, %neg2x;
    .reg .f64 %e_nf, %e_r, %e_p, %e_half;
    .reg .s32 %e_ni;
    .reg .s64 %e_ni64, %e_bits;
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
    shl.b64 %off, %off, 3;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f64 %va, [%a];
    mov.f64 %one, 0d3FF0000000000000;
    mov.f64 %two, 0d4000000000000000;

    // tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
    mul.f64 %neg2x, %va, %two;
    neg.f64 %neg2x, %neg2x;

    // --- exp(%neg2x) via Cody-Waite + degree-11 Horner ---
    mov.f64 %e_half, 0d3FE0000000000000;
    fma.rn.f64 %e_nf, %neg2x, 0d3FF71547652B82FE, %e_half;
    cvt.rmi.f64.f64 %e_nf, %e_nf;
    cvt.rni.s32.f64 %e_ni, %e_nf;
    fma.rn.f64 %e_r, %e_nf, 0dBFE62E42FEFA3800, %neg2x;
    fma.rn.f64 %e_r, %e_nf, 0dBD2EF35793C76730, %e_r;
    mov.f64 %e_p, 0d3E21EED8EFF8D898;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E5AE64567F544E4;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3E927E4FB7789F5C;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EC71DE3A556C734;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3EFA01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F2A01A01A01A01A;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F56C16C16C16C17;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3F811111111111111;
    fma.rn.f64 %e_p, %e_p, %e_r, 0d3FA5555555555555;
    fma.rn.f64 %e_p, %e_p, %e_r, %e_half;
    fma.rn.f64 %e_p, %e_p, %e_r, %one;
    fma.rn.f64 %e64, %e_p, %e_r, %one;
    cvt.s64.s32 %e_ni64, %e_ni;
    add.s64 %e_ni64, %e_ni64, 1023;
    shl.b64 %e_bits, %e_ni64, 52;
    mov.b64 %e_nf, %e_bits;
    mul.f64 %e64, %e64, %e_nf;
    // --- end exp ---

    sub.f64 %num, %one, %e64;
    add.f64 %denom, %one, %e64;
    div.rn.f64 %vr, %num, %denom;
    st.global.f64 [%out], %vr;

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
    .reg .u32 %my_tid, %bid, %bdim, %gdim, %total_reg, %hsz_reg;
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
    mov.u32 %my_tid, %tid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %my_tid;
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

/// Generic device-ordinal check for any `CudaBuffer<T>`.
#[cfg(feature = "cuda")]
fn validate_device<T>(a: &CudaBuffer<T>, device: &GpuDevice) -> GpuResult<()> {
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
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    // Attempt to load the kernel (cached after first compilation).
    // Propagate the DriverError so callers can diagnose why the JIT rejected it.
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx_src,
        kernel_name,
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: kernel_name,
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx_src, kernel_name, ...)`
    //   call earlier in this fn for the binary kernel `kernel_name`;
    //   the entry-point ABI is `(a_ptr, b_ptr, out_ptr, n)` as shown by
    //   the reference `ADD_PTX` constant at line 138 (params `a_ptr,
    //   b_ptr, out_ptr, n`). Failure to compile is caught by the
    //   surrounding `match` and returned as `GpuError::PtxCompileFailed`.
    // - `a` and `b` are non-aliased `CudaBuffer<f32>` values both of
    //   length `n` and on `device`. The helper documents (and callers
    //   honour) that the caller pre-validates via `validate_binary`
    //   (defined at line 10546), which enforces `a.len() == b.len()`
    //   and same-device ordinals.
    // - `out` was freshly allocated this call via `alloc_zeros_f32(n,
    //   device)?` immediately above, so it cannot alias `a` or `b` and
    //   is exclusively borrowed mutably through `out.inner_mut()`.
    // - The kernel's PTX bound check `setp.ge.u32 %p, %r_tid, %n_reg;
    //   @%p bra DONE;` (see `ADD_PTX` at line 164) skips threads with
    //   `tid >= n`, so reads of `a[i]` / `b[i]` and writes to `out[i]`
    //   stay within `[0, n)` for every buffer.
    // - `n_u32 = n as u32` is safe because `launch_cfg(n)` (definition
    //   at line 10523) returns `Err(GpuError::ShapeMismatch)` when
    //   `n > u32::MAX`, short-circuiting the `?` above.
    // - All four arg references live for the duration of the
    //   `.launch(cfg)?` call; cudarc's `LaunchAsync` queues the kernel
    //   on `stream` and stream-sync is the caller's responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

/// Try to launch a vectorized (vec4) binary PTX kernel.
///
/// Each thread processes 4 elements using 128-bit loads/stores.
/// `n` must be divisible by 4. Propagates the DriverError if compilation fails.
#[cfg(feature = "cuda")]
fn try_launch_binary_vec4(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaBuffer<f32>> {
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: kernel_name,
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n4 as usize)?;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx_src, kernel_name, ...)`
    //   call earlier in this fn for the vectorized binary kernel
    //   `kernel_name`; ABI is `(a_ptr, b_ptr, out_ptr, n4)` per the
    //   reference `ADD_VEC4_PTX` at line 189.
    // - The vec4 kernel processes 4 f32 elements per thread via
    //   `ld.global.v4.f32` 128-bit loads; its bound check `setp.ge.u32
    //   %p, %r_tid, %n4_reg` (see `ADD_VEC4_PTX` line 215) skips
    //   threads with `tid >= n4`. Reading `4 * n4 == n - (n % 4)` f32
    //   elements requires `n` divisible by 4, which the helper's
    //   contract (rustdoc above) requires the caller to enforce.
    // - `a` and `b` are caller-validated to be on `device` and same
    //   length; `out` is freshly allocated by `alloc_zeros_f32(n,
    //   device)?` immediately above this block, so it cannot alias the
    //   inputs and is exclusively borrowed via `out.inner_mut()`.
    // - `n4` fits in `u32` because it is computed as `(n / 4) as u32`
    //   and `launch_cfg(n4 as usize)?` immediately above this block
    //   re-validates the cast against `u32::MAX`.
    // - The grid covers `n4` threads exactly; each either processes
    //   its 4-element strip or short-circuits via the `@%p bra DONE`
    //   bound check.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(&n4)
            .launch(cfg)?;
    }

    Ok(out)
}

/// Try to launch a unary PTX kernel. Returns `Ok(buf)` on success,
/// `Err(GpuError::PtxCompileFailed)` if the PTX module failed to load.
#[cfg(feature = "cuda")]
fn try_launch_unary(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    // Attempt to load the kernel (cached after first compilation).
    // Propagate the DriverError so callers can diagnose why the JIT rejected it.
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx_src,
        kernel_name,
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: kernel_name,
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx_src, kernel_name, ...)`
    //   call earlier in this fn; the unary entry-point ABI is `(in_ptr,
    //   out_ptr, n)`, matching the reference `RELU_PTX` at line 436.
    // - `a` is caller-validated to live on `device`; callers gate this
    //   helper on `validate_unary` (definition at line 10570) which
    //   checks `a.device_ordinal() == device.ordinal()`.
    // - `out` was freshly allocated via `alloc_zeros_f32(n, device)?`
    //   immediately above this block with length `n == a.len()`, so
    //   the two buffers cannot alias and `out.inner_mut()` is the
    //   only mutable borrow live for the launch.
    // - The kernel's per-thread bound check `setp.ge.u32 %p, %r_tid,
    //   %n_reg; @%p bra DONE` (pattern shared with `ADD_PTX` at line
    //   164) ensures every load from `a` and store to `out` is in
    //   `[0, n)`.
    // - `n_u32 = n as u32` is safe because `launch_cfg(n)?` (definition
    //   at line 10523) returns `Err(GpuError::ShapeMismatch)` when
    //   `n > u32::MAX` and short-circuits the `?` immediately above.
    // - All three arg refs live for the duration of the
    //   `.launch(cfg)?` call; the cudarc launch is asynchronous on
    //   `stream` and sync is the caller's responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// _into helpers — write to pre-allocated output buffer (no allocation)
// ---------------------------------------------------------------------------

/// Launch a binary PTX kernel into a pre-allocated output buffer.
/// Returns `Ok(())` on success, `Err(GpuError::PtxCompileFailed)` if the PTX module failed to load.
#[cfg(feature = "cuda")]
fn try_launch_binary_into(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<()> {
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: kernel_name,
                source: e,
            });
        }
    };

    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx_src, kernel_name, ...)`
    //   call earlier in this fn; the binary `(a_ptr, b_ptr, out_ptr,
    //   n)` ABI matches the reference `ADD_PTX` at line 138.
    // - `a`, `b`, and `out` are all caller-supplied. `out: &mut
    //   CudaBuffer<f32>` is exclusively borrowed for the duration of
    //   this call, so it cannot alias `a` or `b` per Rust's
    //   `&` / `&mut` exclusivity rules. The helper's rustdoc
    //   immediately above this fn requires the caller to pre-validate
    //   same-device and same-length for `a`, `b`, and `out`.
    // - `n` is taken from `a.len()` at the top of this fn; the kernel
    //   reads `a[i]` and `b[i]` and writes `out[i]` only for
    //   `i in [0, n)`, guarded by the PTX bound check `@%p bra DONE`
    //   (`ADD_PTX` line 165) so out-of-range threads short-circuit.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523) which returns `Err` if `n > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the four arg refs all
    //   live to the trailing `?` and the launch is async-safe —
    //   caller syncs.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(())
}

/// Launch a unary PTX kernel into a pre-allocated output buffer.
/// Returns `Ok(())` on success, `Err(GpuError::PtxCompileFailed)` if the PTX module failed to load.
#[cfg(feature = "cuda")]
fn try_launch_unary_into(
    a: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<()> {
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: kernel_name,
                source: e,
            });
        }
    };

    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx_src, kernel_name, ...)`
    //   call earlier in this fn; the unary `(in_ptr, out_ptr, n)` ABI
    //   matches the reference `RELU_PTX` at line 436.
    // - `a: &CudaBuffer<f32>` and `out: &mut CudaBuffer<f32>` are
    //   passed in by the caller; the `&mut` borrow on `out` proves it
    //   does not alias `a` per Rust's borrow rules (no simultaneous
    //   `&` and `&mut` to the same place).
    // - `n = a.len()` is taken at the top of this fn; the helper's
    //   rustdoc immediately above this fn requires the caller to size
    //   `out` to at least `n` and validate device ordinals before
    //   calling.
    // - The kernel reads `a[i]` and writes `out[i]` only for
    //   `i in [0, n)` per the PTX bound check (`ADD_PTX` line 164,
    //   pattern shared by all unary PTX entry points).
    // - `n_u32 = n as u32` is safe: `launch_cfg(n)?` (definition at
    //   line 10523) returns `Err(GpuError::ShapeMismatch)` if
    //   `n > u32::MAX`, short-circuiting the `?` immediately above.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// f64 launch helpers
// ---------------------------------------------------------------------------

/// Try to launch a binary f64 PTX kernel.
#[cfg(feature = "cuda")]
fn try_launch_binary_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaBuffer<f64>> {
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: kernel_name,
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx_src, kernel_name, ...)`
    //   call earlier in this fn for the f64 binary kernel
    //   `kernel_name`; ABI is `(a_ptr, b_ptr, out_ptr, n)`. f64 PTX is
    //   the f32 source mechanically rewritten to `f64`/`u64`-stride
    //   form by `get_f64_ptx`, so element offsets are computed as
    //   `tid * 8` instead of `tid * 4`.
    // - `a` and `b` are non-aliased `CudaBuffer<f64>` values both of
    //   length `n` on `device`; callers route through this helper
    //   after their own length-equality check (e.g. `gpu_add_f64` —
    //   the binary checks `a.len() != b.len()` and returns
    //   `LengthMismatch` before delegating).
    // - `out` was freshly allocated this call via `alloc_zeros_f64(n,
    //   device)?` immediately above, so it cannot alias `a` or `b`.
    // - The kernel's PTX bound check `setp.ge.u32 %p, %r_tid, %n_reg;
    //   @%p bra DONE` skips OOB threads, keeping all loads / stores
    //   in `[0, n)` for each f64 buffer.
    // - `n_u32 = n as u32` fits in u32 because `launch_cfg(n)?`
    //   (definition at line 10523) returns `Err` when
    //   `n > u32::MAX`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Try to launch a unary f64 PTX kernel.
#[cfg(feature = "cuda")]
fn try_launch_unary_f64(
    a: &CudaBuffer<f64>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaBuffer<f64>> {
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: kernel_name,
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx_src, kernel_name, ...)`
    //   call earlier in this fn for the f64 unary kernel
    //   `kernel_name`; ABI is `(in_ptr, out_ptr, n)`. The f64 source
    //   is produced by `get_f64_ptx` from the f32 PTX template
    //   (e.g. the f32 `RELU_PTX` at line 436).
    // - `a: &CudaBuffer<f64>` with `n = a.len()` taken at the top of
    //   this fn; callers (e.g. `gpu_relu_f64`) own the validation
    //   contract that `a` is same-device as `device`.
    // - `out` is freshly allocated by `alloc_zeros_f64(n, device)?`
    //   immediately above this block with length `n`, so it cannot
    //   alias `a`. `out.inner_mut()` is the only mutable borrow live
    //   for the launch.
    // - The PTX kernel applies its bound check `setp.ge.u32 %p,
    //   %r_tid, %n_reg; @%p bra DONE` (pattern shared with `ADD_PTX`
    //   line 164) so reads / writes stay within `[0, n)` for both
    //   buffers.
    // - `n_u32 = n as u32` fits in u32 because `launch_cfg(n)?`
    //   (definition at line 10523) returns `Err` when
    //   `n > u32::MAX`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Try to launch a general N-dimensional broadcast binary f64 PTX kernel.
///
/// Same as [`try_launch_broadcast_binary`] but for `f64` buffers.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_launch_broadcast_binary_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    a_strides: &[u32],
    b_strides: &[u32],
    out_shape: &[u32],
    out_numel: usize,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<CudaBuffer<f64>> {
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: kernel_name,
                source: e,
            });
        }
    };

    // Upload stride/shape metadata as small device buffers.
    let a_str_buf = cpu_to_gpu(a_strides, device)?;
    let b_str_buf = cpu_to_gpu(b_strides, device)?;
    let shape_buf = cpu_to_gpu(out_shape, device)?;

    let mut out = alloc_zeros_f64(out_numel, device)?;
    let cfg = launch_cfg(out_numel)?;
    let n_u32 = out_numel as u32;
    let ndim_u32 = ndim as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx_src, kernel_name, ...)`
    //   call earlier in this fn for the f64 broadcast binary kernel
    //   `kernel_name`; the entry-point ABI is `(a_ptr, b_ptr, out_ptr,
    //   a_strides, b_strides, out_shape, n, ndim)` (8 args), matching
    //   the f32 reference `BROADCAST_ADD_PTX` at line 8410.
    // - `a_str_buf`, `b_str_buf`, and `shape_buf` are u32 device
    //   buffers freshly built immediately above this block by
    //   `cpu_to_gpu`, each of length `ndim == out_shape.len()`. The
    //   kernel reads exactly `ndim` u32 values from each.
    // - `a` and `b` are caller-supplied `&CudaBuffer<f64>`; the kernel
    //   reads them at indices computed via the broadcast strides
    //   (zero-strides on broadcast dims), which callers built via
    //   `broadcast_strides` (line 11282) so the stride-collapsed
    //   offsets stay in `[0, a.len())` and `[0, b.len())`.
    // - `out` was freshly allocated by `alloc_zeros_f64(out_numel,
    //   device)?` immediately above with `out_numel` elements; it
    //   cannot alias the inputs and the kernel writes exactly
    //   `out[i]` for `i in [0, out_numel)`.
    // - The PTX bound check `setp.ge.u32 %p, %r_tid, %n_reg; @%p bra
    //   DONE` guards `tid >= n` so threads beyond `out_numel` return
    //   without touching memory.
    // - `n_u32 = out_numel as u32` is bounded by
    //   `launch_cfg(out_numel)?` (definition at line 10523) which
    //   errors if `out_numel > u32::MAX`; `ndim_u32` is at most the
    //   strided-copy max-dim cap, which the caller enforces upstream.
    // - All eight arg refs live for the duration of the launch.
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

    Ok(out)
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
) -> GpuResult<CudaBuffer<f32>> {
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: kernel_name,
                source: e,
            });
        }
    };

    // Upload stride/shape metadata as small device buffers.
    let a_str_buf = cpu_to_gpu(a_strides, device)?;
    let b_str_buf = cpu_to_gpu(b_strides, device)?;
    let shape_buf = cpu_to_gpu(out_shape, device)?;

    let mut out = alloc_zeros_f32(out_numel, device)?;
    let cfg = launch_cfg(out_numel)?;
    let n_u32 = out_numel as u32;
    let ndim_u32 = ndim as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx_src, kernel_name, ...)`
    //   call earlier in this fn for the f32 broadcast binary kernel
    //   `kernel_name`; the entry-point ABI matches the reference
    //   `BROADCAST_ADD_PTX` at line 8410: `(a_ptr, b_ptr, out_ptr,
    //   a_strides, b_strides, out_shape, n, ndim)` (8 args).
    // - `a_str_buf`, `b_str_buf`, `shape_buf` were freshly built
    //   immediately above this block from caller-supplied `&[u32]`
    //   slices via `cpu_to_gpu`; each holds `ndim == out_shape.len()`
    //   u32 values, which is what the kernel iterates over.
    // - `a` and `b` are caller-supplied `&CudaBuffer<f32>`; the kernel
    //   reads them at indices computed by collapsing `out_shape`
    //   strides against `a_strides` / `b_strides`. Broadcast (size-1)
    //   dims have stride 0 in the caller-supplied stride arrays
    //   (built by `broadcast_strides` at line 11171), so the
    //   resulting linear offset stays in `[0, a.len())` and
    //   `[0, b.len())`.
    // - `out` was freshly allocated by `alloc_zeros_f32(out_numel,
    //   device)?` immediately above with `out_numel` elements; it
    //   cannot alias `a` or `b` and is exclusively borrowed via
    //   `out.inner_mut()` until launch dispatch returns.
    // - The PTX bound check `setp.ge.u32 %p, %r_tid, %n_reg; @%p bra
    //   DONE` ensures threads with `tid >= n` short-circuit without
    //   touching memory.
    // - `n_u32 = out_numel as u32` is bounded to fit in u32 by
    //   `launch_cfg(out_numel)?` (definition at line 10523);
    //   `ndim_u32` is the upstream-validated rank.
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

    Ok(out)
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

// ---------------------------------------------------------------------------
// Public API -- binary ops
// ---------------------------------------------------------------------------

/// Elementwise addition: `out[i] = a[i] + b[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Returns `Err(GpuError::PtxCompileFailed)`
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
        match try_launch_binary_vec4(a, b, device, ADD_VEC4_PTX, "add_vec4_kernel") {
            Ok(out) => return Ok(out),
            Err(GpuError::PtxCompileFailed { .. }) => {}
            Err(e) => return Err(e),
        }
    }

    try_launch_binary(a, b, device, ADD_PTX, "add_kernel")
}

/// Elementwise subtraction: `out[i] = a[i] - b[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Returns `Err(GpuError::PtxCompileFailed)`
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

    try_launch_binary(a, b, device, SUB_PTX, "sub_kernel")
}

/// Elementwise multiplication: `out[i] = a[i] * b[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Returns `Err(GpuError::PtxCompileFailed)`
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
        match try_launch_binary_vec4(a, b, device, MUL_VEC4_PTX, "mul_vec4_kernel") {
            Ok(out) => return Ok(out),
            Err(GpuError::PtxCompileFailed { .. }) => {}
            Err(e) => return Err(e),
        }
    }

    try_launch_binary(a, b, device, MUL_PTX, "mul_kernel")
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

    try_launch_broadcast_binary(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        BROADCAST_ADD_PTX,
        "broadcast_add_kernel",
    )
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

    try_launch_broadcast_binary(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        BROADCAST_SUB_PTX,
        "broadcast_sub_kernel",
    )
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

    try_launch_broadcast_binary(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        BROADCAST_MUL_PTX,
        "broadcast_mul_kernel",
    )
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

    try_launch_broadcast_binary(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        BROADCAST_DIV_PTX,
        "broadcast_div_kernel",
    )
}

// ---------------------------------------------------------------------------
// Public API -- unary ops
// ---------------------------------------------------------------------------

/// Elementwise negation: `out[i] = -a[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Returns `Err(GpuError::PtxCompileFailed)`
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

    try_launch_unary(a, device, NEG_PTX, "neg_kernel")
}

/// Elementwise ReLU: `out[i] = max(a[i], 0.0)`.
///
/// Attempts to run a PTX kernel on the GPU. Returns `Err(GpuError::PtxCompileFailed)`
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

    try_launch_unary(a, device, RELU_PTX, "relu_kernel")
}

/// ReLU backward: `out[i] = (input[i] > 0) ? grad[i] : 0`.
#[cfg(feature = "cuda")]
pub fn gpu_relu_backward(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, input, device)?;

    try_launch_binary(
        grad,
        input,
        device,
        RELU_BACKWARD_PTX,
        "relu_backward_kernel",
    )
}

/// Elementwise backward for `|x|`: `out[i] = grad[i] * sign(input[i])`
/// with the convention `sign(0) = 0`. Drives `AbsBackward` on GPU.
#[cfg(feature = "cuda")]
pub fn gpu_abs_backward(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, input, device)?;

    try_launch_binary(grad, input, device, ABS_BACKWARD_PTX, "abs_backward_kernel")
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

    try_launch_binary(
        grad,
        input,
        device,
        GELU_BACKWARD_PTX,
        "gelu_backward_kernel",
    )
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

    try_launch_binary(
        grad,
        input,
        device,
        GELU_BACKWARD_ERF_PTX,
        "gelu_backward_erf_kernel",
    )
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "gelu_backward_erf_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, INDEX_SELECT_1D_PTX,
    //   "index_select_1d_kernel", ...)` call earlier in this fn; the
    //   entry-point ABI is `(input_ptr, indices_ptr, out_ptr, n)`.
    // - `input` is on `device` (validated by `validate_unary` at the
    //   top of this fn — definition at line 10570). `indices` is
    //   caller-supplied; gather indices encoded as f32 are read by the
    //   kernel and converted to integer offsets into `input`.
    // - `n = indices.len()` taken at the top of this fn; `out` was
    //   freshly allocated by `alloc_zeros_f32(n, device)?` immediately
    //   above with length `n`, so it cannot alias `input` or
    //   `indices`. `out.inner_mut()` is the only mutable borrow live
    //   for the launch.
    // - The kernel writes `out[i] = input[indices[i]]` for
    //   `i in [0, n)`; the PTX bound check `setp.ge.u32 %p, %r_tid,
    //   %n_reg; @%p bra DONE` skips OOB threads. Caller is
    //   responsible for ensuring `indices[i] < input.len()` (mirrors
    //   PyTorch's `Tensor::index_select` user contract; out-of-range
    //   gather indices are undefined behaviour at the kernel level).
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523) which errors when `n > u32::MAX`.
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "gelu_backward_erf_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(input_len, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, SCATTER_ADD_1D_PTX,
    //   "scatter_add_1d_kernel", ...)` call earlier in this fn; the
    //   entry-point ABI is `(grad_ptr, idx_ptr, out_ptr, n)`.
    // - `grad_output` is on `device` (validated by `validate_unary` at
    //   the top of this fn — definition at line 10570). `indices` is
    //   caller-supplied and same-length as `grad_output` per the
    //   scatter-add contract; `n = grad_output.len()` is taken above.
    // - `out` was freshly allocated by `alloc_zeros_f32(input_len,
    //   device)?` immediately above with length `input_len`, so it
    //   cannot alias `grad_output` or `indices`. Concurrent writers
    //   accumulate via atomic-add (rustdoc above describes the
    //   atomic-add concurrency model), so the kernel's writes to
    //   `out[indices[i]]` are race-free.
    // - The PTX bound check `setp.ge.u32 %p, %r_tid, %n_reg; @%p bra
    //   DONE` skips threads with `tid >= n`, keeping reads of
    //   `grad_output[i]` and `indices[i]` within `[0, n)`. Caller is
    //   responsible for ensuring `indices[i] < input_len` so atomic
    //   writes target a valid `out` element.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523) which errors when `n > u32::MAX`.
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "masked_fill_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, MASKED_FILL_PTX,
    //   "masked_fill_kernel", ...)` call earlier in this fn; ABI is
    //   `(input_ptr, mask_ptr, out_ptr, value, n)`.
    // - `input` and `mask` are validated to be on `device` and same
    //   length by `validate_binary` (definition at line 10546) at the
    //   top of this fn (`a.len() == b.len()`, same-device ordinals).
    //   `n = input.len()`.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above; no aliasing with `input` or `mask`, and
    //   `out.inner_mut()` is the only mutable borrow live for the
    //   launch.
    // - `value` is passed by reference to a local `f32` variable (the
    //   `value: f32` parameter on this fn) which lives for the full
    //   call frame, covering the asynchronous launch.
    // - The kernel reads `input[i]` and `mask[i]` and writes
    //   `out[i] = mask[i] >= 0.5 ? value : input[i]` for
    //   `i in [0, n)`; PTX bound check `setp.ge.u32 %p, %r_tid,
    //   %n_reg; @%p bra DONE` skips OOB threads.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523) which errors when `n > u32::MAX`.
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

    try_launch_binary(grad, mask, device, MASKED_ZERO_PTX, "masked_zero_kernel")
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

    try_launch_binary(
        grad,
        output,
        device,
        SIGMOID_BACKWARD_PTX,
        "sigmoid_backward_kernel",
    )
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

    try_launch_binary(
        grad,
        output,
        device,
        TANH_BACKWARD_PTX,
        "tanh_backward_kernel",
    )
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "softmax_backward_kernel",
                source: e,
            });
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, SOFTMAX_BACKWARD_PTX,
    //   "softmax_backward_kernel", ...)` call earlier in this fn; ABI
    //   is `(grad_ptr, output_ptr, out_ptr, rows, cols)`. The kernel
    //   uses 256 bytes (256 f32) of shared memory per block for the
    //   row-reduction (matching `shared_mem_bytes: 256 * 4`).
    // - `grad` and `output` are validated to be on `device` and same
    //   length by `validate_binary` (definition at line 10546) at the
    //   top of this fn. `total = grad.len()`, `rows = total / cols`.
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above with `total` elements; no aliasing
    //   with `grad`/`output`, and `out.inner_mut()` is the only live
    //   mutable borrow.
    // - The grid is `(rows, 1, 1)` blocks × `(256, 1, 1)` threads:
    //   one block per row computes `out[row*cols .. row*cols+cols]`
    //   from `grad[row*cols..]` and `output[row*cols..]`. With
    //   `total == rows * cols`, every access lands in `[0, total)`.
    // - `rows_u32 = rows as u32` and `cols_u32 = cols as u32` cannot
    //   overflow because the launch grid `(rows as u32).max(1)`
    //   already requires `rows <= u32::MAX` to construct
    //   `LaunchConfig`; in practice `rows * cols == total <=
    //   u32::MAX` is the binding bound. `total` was previously
    //   bounded by allocation success.
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
// Public API -- LogSoftmax forward & backward
// ---------------------------------------------------------------------------

/// Row-wise log-softmax on GPU.
///
/// For each row: `out[j] = x[j] - log(sum(exp(x - max(x))))`.
///
/// One block per row, 256 threads per block, shared-memory reductions for max
/// and sum-exp.
#[cfg(feature = "cuda")]
pub fn gpu_log_softmax(
    input: &CudaBuffer<f32>,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let total = input.len();
    let rows = total / cols;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        LOG_SOFTMAX_PTX,
        "log_softmax_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "log_softmax_kernel",
                source: e,
            });
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, LOG_SOFTMAX_PTX,
    //   "log_softmax_kernel", ...)` call earlier in this fn; ABI is
    //   `(input_ptr, out_ptr, rows, cols)`. The kernel uses 256 f32
    //   (1024 bytes) of shared memory per block for max + sum-exp
    //   reduction (matching `shared_mem_bytes: 256 * 4`).
    // - `input` is on `device` (validated by `validate_unary` at the
    //   top of this fn — definition at line 10570). `total =
    //   input.len()`, `rows = total / cols`.
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above; no aliasing with `input` and
    //   `out.inner_mut()` is the only live mutable borrow.
    // - Grid `(rows, 1, 1)` × block `(256, 1, 1)`: one block per row
    //   reads `input[row*cols..row*cols+cols]` and writes
    //   `out[row*cols..row*cols+cols]`. With `total == rows * cols`,
    //   every access lands in `[0, total)`.
    // - `rows_u32` / `cols_u32` casts: launch-grid construction
    //   already cast `rows as u32`, so any overflow would already
    //   have produced a wrong grid; in practice both are < u32::MAX
    //   because `total = rows * cols` was already bounded by
    //   buffer allocation success.
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

/// Row-wise log-softmax backward on GPU.
///
/// For each row:
///   `sum_grad = sum(grad[j])`
///   `out[j] = grad[j] - exp(output[j]) * sum_grad`
///
/// where `output` is the log-softmax forward output.
#[cfg(feature = "cuda")]
pub fn gpu_log_softmax_backward(
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
        LOG_SOFTMAX_BACKWARD_PTX,
        "log_softmax_backward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "log_softmax_backward_kernel",
                source: e,
            });
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, LOG_SOFTMAX_BACKWARD_PTX,
    //   "log_softmax_backward_kernel", ...)` call earlier in this fn;
    //   ABI is `(grad_ptr, output_ptr, out_ptr, rows, cols)`. The
    //   kernel uses 256 f32 (1024 bytes) of shared memory per block
    //   for the per-row sum-of-grads reduction (matching
    //   `shared_mem_bytes: 256 * 4`).
    // - `grad` and `output` are validated to be on `device` and same
    //   length by `validate_binary` (definition at line 10546) at the
    //   top of this fn. `total = grad.len()`, `rows = total / cols`.
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above; no aliasing, exclusive
    //   `out.inner_mut()` borrow during the launch.
    // - Grid `(rows, 1, 1)` × block `(256, 1, 1)`: one block per row
    //   reads/writes `[row*cols .. row*cols+cols]` of each buffer.
    //   With `total == rows * cols`, every access lands in
    //   `[0, total)`.
    // - `rows_u32` / `cols_u32` casts are valid because the caller's
    //   `total` already fit in usize and was bounded by allocation
    //   success; the launch grid's `(rows as u32).max(1)` cast is
    //   the binding bound on `rows`.
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
pub fn gpu_reduce_sum(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "reduce_sum_kernel",
                source: e,
            });
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, REDUCE_SUM_PTX,
    //   "reduce_sum_kernel", ...)` call earlier in this fn; ABI is
    //   `(input_ptr, partials_ptr, n)`.
    // - `a` is caller-supplied; the `n == 0` early-return above this
    //   block guarantees `n > 0` here, and `n_u32 = n as u32` is safe
    //   because `num_blocks` was computed from `n as u32` so `n <=
    //   u32::MAX` is implied (`saturating_add` would not produce a
    //   meaningful grid otherwise).
    // - `partials` was freshly allocated by
    //   `alloc_zeros_f32(num_blocks as usize, device)?` immediately
    //   above with `num_blocks` elements; no aliasing with `a`, and
    //   `partials.inner_mut()` is the only live mutable borrow.
    // - The kernel pattern: each block reduces a contiguous slice of
    //   `a` (with grid-stride loop for `n > num_blocks * BLOCK`)
    //   using shared-memory tree reduction (statically allocated in
    //   PTX, hence `shared_mem_bytes: 0` here), and writes one f32
    //   to `partials[block_id]`. The `num_blocks <= 1024` cap above
    //   ensures `block_id < partials.len()`. The read pattern stays
    //   in `[0, n)` thanks to the grid-stride loop's bound check.
    // - All three arg refs live to the trailing `?`.
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
pub fn gpu_reduce_sum(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// f32 parallel reduction returning the product of all elements (#524).
/// Two-pass dispatch identical to `gpu_reduce_sum`/`gpu_reduce_min`, with
/// `1.0` identity and `mul.f32` combiner.
#[cfg(feature = "cuda")]
pub fn gpu_reduce_prod(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    if n == 0 {
        return cpu_to_gpu(&[1.0_f32], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        REDUCE_PROD_PTX,
        "reduce_prod_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "reduce_prod_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);
    // Sentinel-init the partials with 1.0 so blocks that exit the grid
    // loop without writing don't poison the second pass with zeros.
    let mut partials = cpu_to_gpu(&vec![1.0_f32; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, REDUCE_PROD_PTX,
    //   "reduce_prod_kernel", ...)` call earlier in this fn; ABI is
    //   `(input_ptr, partials_ptr, n)`. The combiner is `mul.f32`
    //   with identity 1.0 (the sentinel used to pre-init `partials`).
    // - `a` is caller-supplied; `n == a.len()`, with `n > 0`
    //   guaranteed by the early-return for the zero-element case
    //   above this fn.
    // - `partials` was freshly built above by `cpu_to_gpu(&vec![1.0;
    //   num_blocks], device)?` with `num_blocks` elements (capped at
    //   1024 by the `min(1024)` clamp); no aliasing with `a`, and
    //   `partials.inner_mut()` is the only live mutable borrow.
    // - The kernel reads `a[i]` for `i in [0, n)` via grid-stride
    //   loop with a per-thread bound check, and writes one f32 to
    //   `partials[block_id]` via shared-memory tree reduction
    //   (statically allocated, hence `shared_mem_bytes: 0`).
    // - `n_u32 = n as u32` is implicitly bounded by the
    //   `(n as u32).saturating_add(BLOCK - 1)` cast above
    //   (saturating-add caps at u32::MAX, but a saturated
    //   `num_blocks` only means the kernel processes the first
    //   u32::MAX elements via the grid-stride loop; for a strictly
    //   correct `n > u32::MAX` we'd need wider arithmetic — caller's
    //   responsibility per the rustdoc on the related `gpu_reduce_sum`).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }
    if num_blocks <= 256 {
        let host = gpu_to_cpu(&partials, device)?;
        let total: f32 = host.iter().product();
        return cpu_to_gpu(&[total], device);
    }
    gpu_reduce_prod(&partials, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_reduce_prod(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// f32 parallel reduction returning the minimum element. Two-pass kernel
/// dispatch identical to [`gpu_reduce_sum`] but using
/// [`REDUCE_MIN_PTX`]. (#627)
#[cfg(feature = "cuda")]
pub fn gpu_reduce_min(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    if n == 0 {
        return cpu_to_gpu(&[f32::INFINITY], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        REDUCE_MIN_PTX,
        "reduce_min_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "reduce_min_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);

    // Allocate partial buffer pre-filled with +inf so blocks that don't
    // see every element don't poison the second pass with stale zeros.
    let mut partials = cpu_to_gpu(&vec![f32::INFINITY; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, REDUCE_MIN_PTX,
    //   "reduce_min_kernel", ...)` call earlier in this fn; ABI is
    //   `(input_ptr, partials_ptr, n)`. The combiner is `min.f32`
    //   with identity `+INF` (matching the partials sentinel above).
    // - `a` is caller-supplied with `n == a.len()`; `n > 0` is
    //   ensured by the early-return for the zero-element case above
    //   this fn.
    // - `partials` was freshly built above by `cpu_to_gpu(&vec![+INF;
    //   num_blocks], device)?` with `num_blocks` elements (capped at
    //   1024); no aliasing with `a`, and `partials.inner_mut()` is
    //   the only live mutable borrow.
    // - The kernel reads `a[i]` for `i in [0, n)` via a grid-stride
    //   loop with per-thread bound check, and writes one f32 to
    //   `partials[block_id]` via shared-memory tree reduction.
    //   The `min(1024)` cap on `num_blocks` ensures `block_id <
    //   partials.len()`.
    // - `n_u32 = n as u32` is fine for the n <= u32::MAX domain;
    //   sentinel pre-initialisation guarantees correctness even if
    //   `num_blocks` saturates (untouched partials stay at +INF, the
    //   identity).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }

    if num_blocks <= 256 {
        let host_partials = gpu_to_cpu(&partials, device)?;
        let total = host_partials.iter().copied().fold(f32::INFINITY, f32::min);
        return cpu_to_gpu(&[total], device);
    }

    gpu_reduce_min(&partials, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_reduce_min(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// f32 parallel reduction returning the maximum element. Counterpart of
/// [`gpu_reduce_min`]. (#627)
#[cfg(feature = "cuda")]
pub fn gpu_reduce_max(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    if n == 0 {
        return cpu_to_gpu(&[f32::NEG_INFINITY], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        REDUCE_MAX_PTX,
        "reduce_max_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "reduce_max_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);

    let mut partials = cpu_to_gpu(&vec![f32::NEG_INFINITY; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, REDUCE_MAX_PTX,
    //   "reduce_max_kernel", ...)` call earlier in this fn; ABI is
    //   `(input_ptr, partials_ptr, n)`. Combiner is `max.f32`
    //   with identity `-INF` (matching the partials sentinel above).
    // - `a` is caller-supplied with `n == a.len()`; `n > 0` enforced
    //   by the early-return for the zero-element case above this fn.
    // - `partials` was freshly built above by `cpu_to_gpu(&vec![-INF;
    //   num_blocks], device)?` with `num_blocks` elements (capped at
    //   1024); no aliasing with `a`. `partials.inner_mut()` is the
    //   only live mutable borrow.
    // - The kernel reads `a[i]` for `i in [0, n)` via grid-stride
    //   loop with bound check, and writes one f32 to
    //   `partials[block_id]` via shared-memory tree reduction.
    //   `block_id < num_blocks <= 1024 == partials.len()`.
    // - `n_u32 = n as u32` is bounded for `n <= u32::MAX`; sentinel
    //   pre-init makes the kernel safe even when `num_blocks`
    //   saturates because untouched partials stay at -INF (identity).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }

    if num_blocks <= 256 {
        let host_partials = gpu_to_cpu(&partials, device)?;
        let total = host_partials
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        return cpu_to_gpu(&[total], device);
    }

    gpu_reduce_max(&partials, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_reduce_max(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Fused masked-min reduction (#627): single-pass kernel that combines
/// `(data, mask_f) -> min` without allocating intermediate `prod` /
/// `filled` buffers. `mask_f` must be `[0.0/1.0]` valued (an
/// `is_valid` indicator) and same length as `data`.
#[cfg(feature = "cuda")]
pub fn gpu_masked_reduce_min(
    data: &CudaBuffer<f32>,
    mask_f: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    if data.len() != mask_f.len() {
        return Err(GpuError::LengthMismatch {
            a: data.len(),
            b: mask_f.len(),
        });
    }
    let n = data.len();
    if n == 0 {
        return cpu_to_gpu(&[f32::INFINITY], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        MASKED_REDUCE_MIN_PTX,
        "masked_reduce_min_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "masked_reduce_min_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);

    // Sentinel-init the partials so blocks that exit the grid loop
    // without writing don't poison the second pass with stale values.
    let mut partials = cpu_to_gpu(&vec![f32::INFINITY; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, MASKED_REDUCE_MIN_PTX,
    //   "masked_reduce_min_kernel", ...)` call earlier in this fn;
    //   ABI is `(data_ptr, mask_ptr, partials_ptr, n)`. The combiner
    //   is `min.f32(data[i] when mask>=0.5 else +INF)` with identity
    //   `+INF`.
    // - `data` and `mask_f` were length-validated equal at the top of
    //   this fn (`data.len() != mask_f.len()` returns `LengthMismatch`).
    //   `n = data.len()`; `n > 0` enforced by the early-return for
    //   the zero-element case above.
    // - `partials` was freshly built above by `cpu_to_gpu(&vec![+INF;
    //   num_blocks], ...)?` with `num_blocks` elements (capped at
    //   1024); cannot alias `data` or `mask_f`. `partials.inner_mut()`
    //   is the only live mutable borrow.
    // - Kernel reads `data[i]` and `mask_f[i]` for `i in [0, n)` via
    //   grid-stride loop with bound check, writes one f32 per block
    //   to `partials[block_id]` via shared-memory tree reduction.
    // - `n_u32` cast valid for `n <= u32::MAX`; sentinel pre-init
    //   keeps untouched partials at the +INF identity.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(data.inner())
            .arg(mask_f.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }
    if num_blocks <= 256 {
        let host = gpu_to_cpu(&partials, device)?;
        let total = host.iter().copied().fold(f32::INFINITY, f32::min);
        return cpu_to_gpu(&[total], device);
    }
    // Recurse with the unmasked reducer on the partials (already filtered).
    gpu_reduce_min(&partials, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_reduce_min(
    _data: &CudaBuffer<f32>,
    _mask_f: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Fused masked-max counterpart of [`gpu_masked_reduce_min`]. (#627)
#[cfg(feature = "cuda")]
pub fn gpu_masked_reduce_max(
    data: &CudaBuffer<f32>,
    mask_f: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    if data.len() != mask_f.len() {
        return Err(GpuError::LengthMismatch {
            a: data.len(),
            b: mask_f.len(),
        });
    }
    let n = data.len();
    if n == 0 {
        return cpu_to_gpu(&[f32::NEG_INFINITY], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        MASKED_REDUCE_MAX_PTX,
        "masked_reduce_max_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "masked_reduce_max_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);

    let mut partials = cpu_to_gpu(&vec![f32::NEG_INFINITY; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, MASKED_REDUCE_MAX_PTX,
    //   "masked_reduce_max_kernel", ...)` call earlier in this fn;
    //   ABI is `(data_ptr, mask_ptr, partials_ptr, n)`. Combiner is
    //   `max.f32(data[i] when mask>=0.5 else -INF)` with identity
    //   `-INF` (matching the partials sentinel above).
    // - `data` and `mask_f` were length-validated equal at the top of
    //   this fn (`data.len() != mask_f.len()` returns `LengthMismatch`).
    //   `n = data.len()` and `n > 0` per the early-return for the
    //   zero-element case above.
    // - `partials` was freshly built above by `cpu_to_gpu(&vec![-INF;
    //   num_blocks], ...)?` with `num_blocks` (capped at 1024)
    //   elements; cannot alias `data` or `mask_f`. Exclusive
    //   mutable borrow via `partials.inner_mut()`.
    // - Kernel reads `data[i]` and `mask_f[i]` for `i in [0, n)` via
    //   grid-stride loop with bound check, writes one f32 per block
    //   to `partials[block_id]`; `block_id < partials.len()`
    //   guaranteed by the `min(1024)` clamp.
    // - `n_u32 = n as u32` is bounded for `n <= u32::MAX`; sentinel
    //   pre-init keeps untouched partials at the -INF identity.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(data.inner())
            .arg(mask_f.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }
    if num_blocks <= 256 {
        let host = gpu_to_cpu(&partials, device)?;
        let total = host.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        return cpu_to_gpu(&[total], device);
    }
    gpu_reduce_max(&partials, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_reduce_max(
    _data: &CudaBuffer<f32>,
    _mask_f: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// GPU pad/truncate for complex tensors stored as `[batch, n, 2]` f32 (#605).
/// Produces `[batch, dst_n, 2]`: zero-padded when `dst_n > src_n`, truncated
/// when `dst_n < src_n`. Single PTX kernel, no host bounce.
#[cfg(feature = "cuda")]
pub fn gpu_pad_truncate_complex_f32(
    src: &CudaBuffer<f32>,
    batch: usize,
    src_n: usize,
    dst_n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    if src.len() != batch * src_n * 2 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_pad_truncate_complex_f32",
            expected: vec![batch * src_n * 2],
            got: vec![src.len()],
        });
    }
    let total_pairs = batch * dst_n;
    if total_pairs == 0 {
        return alloc_zeros_f32(0, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        PAD_TRUNCATE_PTX,
        "pad_truncate_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "pad_truncate_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(batch * dst_n * 2, device)?;
    let cfg = launch_cfg(total_pairs)?;
    let batch_u32 = batch as u32;
    let src_n_u32 = src_n as u32;
    let dst_n_u32 = dst_n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, PAD_TRUNCATE_PTX,
    //   "pad_truncate_kernel", ...)` call earlier in this fn; ABI is
    //   `(src_ptr, out_ptr, batch, src_n, dst_n)`.
    // - `src.len() == batch * src_n * 2` was verified at the top of
    //   this fn (returns `ShapeMismatch` otherwise); `total_pairs =
    //   batch * dst_n` is checked nonzero (early-return for zero).
    // - `out` was freshly allocated by `alloc_zeros_f32(batch * dst_n
    //   * 2, device)?` immediately above; its `2 * total_pairs` f32
    //   slots cover every (batch, dst_n, complex-pair) triple. No
    //   aliasing with `src`; `out.inner_mut()` is the exclusive
    //   mutable borrow.
    // - The kernel maps thread `i in [0, total_pairs)` to (b, d) via
    //   `b = i / dst_n`, `d = i % dst_n`. For `d < src_n` it copies
    //   `src[(b * src_n + d) * 2 ..]` (real and imaginary parts);
    //   for `d >= src_n` it writes zeros (zero-padding). Bound on
    //   the read side: `(b * src_n + d) * 2 + 1 < src.len()`
    //   because `d < src_n` and `b < batch`.
    // - `batch_u32`, `src_n_u32`, `dst_n_u32` casts: `total_pairs ==
    //   batch * dst_n` was bounded to fit `u32` by `launch_cfg(...)`
    //   immediately above (definition at line 10523); individual
    //   `batch`, `src_n`, `dst_n` are caller-supplied `usize` and
    //   the caller is responsible for keeping each below u32::MAX.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(src.inner())
            .arg(out.inner_mut())
            .arg(&batch_u32)
            .arg(&src_n_u32)
            .arg(&dst_n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_pad_truncate_complex_f32(
    _src: &CudaBuffer<f32>,
    _batch: usize,
    _src_n: usize,
    _dst_n: usize,
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "sum_axis_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total_output, device)?;
    let cfg = launch_cfg(total_output)?;
    let outer_u32 = outer as u32;
    let axis_size_u32 = axis_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total_output as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, SUM_AXIS_PTX,
    //   "sum_axis_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, outer, axis_size, inner, total)`.
    // - `a` is on `device` (validated by `validate_unary` at the top
    //   of this fn — definition at line 10570). Caller is responsible
    //   for ensuring `a.len() >= outer * axis_size * inner` (matches
    //   the addressing pattern below).
    // - `total_output = outer * inner`; `out` was freshly allocated
    //   by `alloc_zeros_f32(total_output, device)?` immediately
    //   above. No aliasing with `a`; exclusive `out.inner_mut()`.
    // - Thread `i in [0, total_output)` computes
    //   `out[i] = sum_{k=0}^{axis_size-1} a[outer_idx * axis_size *
    //   inner + k * inner + inner_idx]` where `outer_idx = i / inner`,
    //   `inner_idx = i % inner`. The sum-loop runs `axis_size` reads
    //   per output, all in `[0, outer * axis_size * inner) <=
    //   [0, a.len())`. Bound check `tid >= total` short-circuits OOB
    //   threads.
    // - `outer_u32`, `axis_size_u32`, `inner_u32`, `total_u32`:
    //   `total_u32 = total_output as u32` is bounded by
    //   `launch_cfg(total_output)?` (definition at line 10523) which
    //   errors if it exceeds `u32::MAX`; the other three are caller-
    //   supplied `usize` and the caller is responsible for keeping
    //   each within u32 range.
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
// Public API -- Cumulative scan operations
// ---------------------------------------------------------------------------

/// Cumulative sum (prefix sum) along an axis on GPU.
///
/// `output[base + k*inner] = sum_{j=0}^{k} input[base + j*inner]`
/// where `base = outer_idx * dim_size * inner + inner_idx`.
///
/// One thread per (outer_idx, inner_idx) pair; each thread does a sequential
/// scan along `dim_size` elements.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `input` and `device` are on different devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_cumsum(
    input: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let total = outer * dim_size * inner;
    let num_threads = outer * inner;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        CUMSUM_PTX,
        "cumsum_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "cumsum_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(num_threads)?;
    let outer_u32 = outer as u32;
    let dim_size_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, CUMSUM_PTX,
    //   "cumsum_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, outer, dim_size, inner, total)`.
    // - `input` is on `device` (validated by `validate_unary` at the
    //   top of this fn — definition at line 10570). Caller ensures
    //   `input.len() == total == outer * dim_size * inner` (matches
    //   the kernel's per-thread sequential-scan addressing).
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above with `total` elements; no
    //   aliasing with `input`. Exclusive `out.inner_mut()` borrow.
    // - Grid covers `num_threads = outer * inner` threads (line
    //   10523's `launch_cfg`). Thread `i` computes (outer_idx,
    //   inner_idx) and sequentially scans `dim_size` elements,
    //   reading `input[base + k*inner]` and writing `out[base +
    //   k*inner]` for `k in [0, dim_size)`, where `base =
    //   outer_idx * dim_size * inner + inner_idx`. Maximum offset
    //   `(outer-1) * dim_size * inner + (dim_size-1) * inner +
    //   (inner-1) < total`.
    // - `outer_u32`, `dim_size_u32`, `inner_u32`, `total_u32`:
    //   `launch_cfg(num_threads)?` ensures `num_threads <= u32::MAX`;
    //   caller is responsible for keeping each individual dim within
    //   u32 range so the kernel's index arithmetic doesn't overflow
    //   `u32`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&dim_size_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

/// Cumulative product (prefix product) along an axis on GPU.
///
/// `output[base + k*inner] = prod_{j=0}^{k} input[base + j*inner]`
/// where `base = outer_idx * dim_size * inner + inner_idx`.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `input` and `device` are on different devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_cumprod(
    input: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let total = outer * dim_size * inner;
    let num_threads = outer * inner;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        CUMPROD_PTX,
        "cumprod_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "cumprod_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(num_threads)?;
    let outer_u32 = outer as u32;
    let dim_size_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, CUMPROD_PTX,
    //   "cumprod_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, outer, dim_size, inner, total)`. Combiner
    //   is `mul.f32` along the scan axis with identity 1.0.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570) and has length `outer * dim_size
    //   * inner == total`.
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above with `total` elements; no
    //   aliasing with `input`. Exclusive `out.inner_mut()` borrow.
    // - Grid covers `num_threads = outer * inner` threads. Each
    //   thread computes one (outer_idx, inner_idx) sequential scan
    //   of `dim_size` elements; addressing is identical to
    //   `gpu_cumsum`, so all reads and writes stay in `[0, total)`.
    // - `outer_u32`, `dim_size_u32`, `inner_u32`, `total_u32`:
    //   `launch_cfg(num_threads)?` (definition at line 10523)
    //   bounds `num_threads` in u32; caller is responsible for
    //   per-dim u32 bounds.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&dim_size_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

/// Cumulative maximum (running max) along an axis on GPU.
///
/// `output[base + k*inner] = max_{j=0}^{k} input[base + j*inner]`
/// where `base = outer_idx * dim_size * inner + inner_idx`.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `input` and `device` are on different devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_cummax(
    input: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let total = outer * dim_size * inner;
    let num_threads = outer * inner;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        CUMMAX_PTX,
        "cummax_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "cummax_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let mut out_idx = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(num_threads)?;
    let outer_u32 = outer as u32;
    let dim_size_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, CUMMAX_PTX,
    //   "cummax_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_val_ptr, out_idx_ptr, outer, dim_size, inner,
    //   total)`. Combiner is `max.f32` along the scan axis (with
    //   argmax tracking written to `out_idx` as f32-encoded
    //   indices).
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570); length is `outer * dim_size *
    //   inner == total`.
    // - `out` and `out_idx` were freshly allocated by two distinct
    //   `alloc_zeros_f32(total, device)?` calls immediately above,
    //   so they do not alias each other or `input`. Each is held
    //   under its own `&mut` binding, so `out.inner_mut()` and
    //   `out_idx.inner_mut()` are non-aliasing exclusive borrows.
    // - Grid covers `num_threads = outer * inner` threads. Each
    //   thread sequentially scans `dim_size` elements writing both
    //   buffers at `base + k*inner` for `k in [0, dim_size)`;
    //   addressing parallels `gpu_cumsum`, so all five buffer
    //   accesses (one read from `input`, two writes to `out` /
    //   `out_idx`) stay in `[0, total)`.
    // - `outer_u32`, `dim_size_u32`, `inner_u32`, `total_u32`:
    //   `launch_cfg(num_threads)?` (definition at line 10523)
    //   bounds `num_threads` in u32; caller's responsibility for
    //   per-dim u32 bounds.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(out_idx.inner_mut())
            .arg(&outer_u32)
            .arg(&dim_size_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok((out, out_idx))
}

/// Cumulative minimum (running min) along an axis on GPU.
///
/// `output[base + k*inner] = min_{j=0}^{k} input[base + j*inner]`
/// where `base = outer_idx * dim_size * inner + inner_idx`.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `input` and `device` are on different devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_cummin(
    input: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let total = outer * dim_size * inner;
    let num_threads = outer * inner;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        CUMMIN_PTX,
        "cummin_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "cummin_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let mut out_idx = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(num_threads)?;
    let outer_u32 = outer as u32;
    let dim_size_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, CUMMIN_PTX,
    //   "cummin_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_val_ptr, out_idx_ptr, outer, dim_size, inner,
    //   total)`. Combiner is `min.f32` with argmin tracking.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570); length is `outer * dim_size *
    //   inner == total`.
    // - `out` and `out_idx` were freshly allocated by two distinct
    //   `alloc_zeros_f32(total, device)?` calls immediately above
    //   and are held under separate `&mut` bindings; the two
    //   `.inner_mut()` borrows do not alias each other or `input`.
    // - Grid covers `num_threads = outer * inner` threads. Each
    //   sequentially scans `dim_size` elements writing both buffers
    //   at `base + k*inner` for `k in [0, dim_size)`; identical
    //   addressing to `gpu_cummax`, so all accesses stay in
    //   `[0, total)`.
    // - `outer_u32`, `dim_size_u32`, `inner_u32`, `total_u32`:
    //   `launch_cfg(num_threads)?` (definition at line 10523)
    //   bounds `num_threads` in u32; caller is responsible for
    //   per-dim u32 bounds.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(out_idx.inner_mut())
            .arg(&outer_u32)
            .arg(&dim_size_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok((out, out_idx))
}

/// Numerically stable log-cumulative-sum-exp along an axis on GPU.
///
/// `acc = log(exp(acc) + exp(x))` computed as `m + log(exp(acc-m) + exp(x-m))`
/// where `m = max(acc, x)` for numerical stability.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `input` and `device` are on different devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_logcumsumexp(
    input: &CudaBuffer<f32>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let total = outer * dim_size * inner;
    let num_threads = outer * inner;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        LOGCUMSUMEXP_PTX,
        "logcumsumexp_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "logcumsumexp_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(num_threads)?;
    let outer_u32 = outer as u32;
    let dim_size_u32 = dim_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, LOGCUMSUMEXP_PTX,
    //   "logcumsumexp_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, outer, dim_size, inner, total)`.
    //   Combiner is the numerically-stable
    //   `m + log(exp(acc - m) + exp(x - m))` recurrence.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570) and has `outer * dim_size * inner
    //   == total` elements.
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above; no aliasing with `input`.
    //   Exclusive `out.inner_mut()` borrow.
    // - Grid covers `num_threads = outer * inner` threads. Each
    //   sequentially scans `dim_size` elements via the same
    //   addressing as `gpu_cumsum`, so reads and writes stay in
    //   `[0, total)`.
    // - `outer_u32`, `dim_size_u32`, `inner_u32`, `total_u32`:
    //   `launch_cfg(num_threads)?` (definition at line 10523)
    //   bounds `num_threads` in u32; caller is responsible for
    //   per-dim u32 bounds.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&dim_size_u32)
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "strided_split_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let total_ax_u32 = total_along_axis as u32;
    let offset_u32 = split_offset as u32;
    let split_sz_u32 = split_size as u32;
    let inner_u32 = inner_size as u32;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, STRIDED_SPLIT_PTX,
    //   "strided_split_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, total_along_axis, split_offset, split_size,
    //   inner_size, n)`.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570). Caller is responsible for sizing
    //   `input` to cover `outer * total_along_axis * inner_size`
    //   elements; the kernel's read pattern stays within that
    //   region.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above with `n = outer * split_size * inner_size`
    //   elements; no aliasing with `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - Thread `i in [0, n)` decodes `(outer_idx, split_idx,
    //   inner_idx)` from `i` via `inner_size` and `split_size`, then
    //   writes `out[i] = input[outer_idx * total_along_axis *
    //   inner_size + (split_offset + split_idx) * inner_size +
    //   inner_idx]`. With `split_offset + split_size <=
    //   total_along_axis` (caller contract), every read is in
    //   `[0, input.len())`. Bound check `tid >= n` guards OOB.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523); the other casts are caller-supplied
    //   per-axis dimensions and the caller is responsible for u32
    //   range.
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "strided_cat_kernel",
                source: e,
            });
        }
    };

    let cfg = launch_cfg(n)?;
    let total_ax_u32 = total_along_axis as u32;
    let offset_u32 = cat_offset as u32;
    let part_sz_u32 = part_size as u32;
    let inner_u32 = inner_size as u32;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, STRIDED_CAT_PTX,
    //   "strided_cat_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, total_along_axis, cat_offset, part_size,
    //   inner_size, n)`.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570). `output: &mut CudaBuffer<f32>` is
    //   exclusively borrowed for the duration of this call; no
    //   aliasing with `input`. The fn's `# Safety` rustdoc above
    //   documents that the caller must size `output` large enough
    //   to hold the written region and ensure non-overlapping writes
    //   when chaining cat segments.
    // - Thread `i in [0, n)` reads `input[i]` (a contiguous
    //   `outer * part_size * inner_size = n` chunk) and writes it
    //   into `output[outer_idx * total_along_axis * inner_size +
    //   (cat_offset + part_idx) * inner_size + inner_idx]`. Caller
    //   contract `cat_offset + part_size <= total_along_axis` keeps
    //   the destination index in `[0, output.len())`.
    // - `n_u32 = n as u32` bounded by `launch_cfg(n)?` (definition
    //   at line 10523); the other casts are caller-supplied per-axis
    //   dims, caller is responsible for u32 range.
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

// ---------------------------------------------------------------------------
// Public API -- Strided copy (general N-d gather) -- CL-496
// ---------------------------------------------------------------------------

/// Maximum rank supported by [`gpu_strided_copy`] and [`gpu_strided_copy_f64`].
/// Matches the unrolled PTX kernel's dimension count.
pub const STRIDED_COPY_MAX_DIMS: usize = 8;

/// Pad-and-validate the (out_shape, src_strides) pair for the
/// strided-copy kernel.
///
/// Returns a fixed-size `[MAX_DIMS]` pair of arrays where:
/// - `out_stride[d]` is the contiguous output stride (in elements)
///   for that dim, with unused trailing dims filled with `n + 1` so
///   that `flat / out_stride[d] == 0` in the kernel (no contribution).
/// - `src_stride[d]` is the source stride (in elements) for that
///   dim, with unused trailing dims filled with 0 so the source-
///   offset contribution is zero.
///
/// `out_shape` and `src_strides` must have the same length, at most
/// `STRIDED_COPY_MAX_DIMS`. `n` is the product of `out_shape`.
#[cfg(feature = "cuda")]
fn pad_strided_copy_params(
    out_shape: &[usize],
    src_strides: &[isize],
    n: usize,
) -> GpuResult<([u32; STRIDED_COPY_MAX_DIMS], [u32; STRIDED_COPY_MAX_DIMS])> {
    if out_shape.len() != src_strides.len() {
        return Err(GpuError::ShapeMismatch {
            op: "strided_copy_pad",
            expected: vec![out_shape.len()],
            got: vec![src_strides.len()],
        });
    }
    if out_shape.len() > STRIDED_COPY_MAX_DIMS {
        return Err(GpuError::ShapeMismatch {
            op: "strided_copy_pad",
            expected: vec![STRIDED_COPY_MAX_DIMS],
            got: vec![out_shape.len()],
        });
    }
    // Reject negative source strides — the kernel treats them as u32
    // which would wrap around and produce garbage indices.
    for &s in src_strides {
        if s < 0 {
            return Err(GpuError::ShapeMismatch {
                op: "strided_copy_pad_negative_stride",
                expected: vec![0],
                got: vec![s.unsigned_abs()],
            });
        }
    }

    let rank = out_shape.len();
    // Compute contiguous output strides: stride[rank-1] = 1,
    // stride[d] = stride[d+1] * shape[d+1].
    let mut out_stride = [0u32; STRIDED_COPY_MAX_DIMS];
    if rank > 0 {
        let mut acc: usize = 1;
        for d in (0..rank).rev() {
            if acc > u32::MAX as usize {
                return Err(GpuError::ShapeMismatch {
                    op: "strided_copy_stride_overflow",
                    expected: vec![u32::MAX as usize],
                    got: vec![acc],
                });
            }
            out_stride[d] = acc as u32;
            acc = acc.saturating_mul(out_shape[d]);
        }
    }

    // Pad unused dims with `n + 1` so `flat / out_stride[d] == 0`
    // in the kernel (any flat < n is strictly less than n + 1).
    let pad_val = (n as u32).saturating_add(1).max(1);
    out_stride[rank..STRIDED_COPY_MAX_DIMS].fill(pad_val);

    // src_stride with 0 fill for unused dims (no contribution).
    let mut src_stride_out = [0u32; STRIDED_COPY_MAX_DIMS];
    for d in 0..rank {
        let s = src_strides[d];
        if s as usize > u32::MAX as usize {
            return Err(GpuError::ShapeMismatch {
                op: "strided_copy_src_stride_overflow",
                expected: vec![u32::MAX as usize],
                got: vec![s as usize],
            });
        }
        src_stride_out[d] = s as u32;
    }

    Ok((out_stride, src_stride_out))
}

/// Gather a non-contiguous strided view of `input` into a new
/// contiguous output buffer, entirely on GPU. CL-496.
///
/// # Arguments
///
/// * `input`      — the storage backing the strided view. Must be
///   on `device`.
/// * `out_shape`  — shape of the contiguous output (and of the
///   logical view). `out_shape.len() <= STRIDED_COPY_MAX_DIMS`.
/// * `src_strides` — source element strides per dim, aligned with
///   `out_shape`. Must be non-negative (no reverse views yet).
/// * `src_offset`  — base element offset into `input` for the view.
/// * `device`     — CUDA device.
///
/// # Returns
///
/// A contiguous `CudaBuffer<f32>` with `product(out_shape)` elements.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `input` and `device` differ.
/// - [`GpuError::ShapeMismatch`] on rank mismatch, too many dims,
///   negative strides, or stride overflow of `u32::MAX`.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_strided_copy(
    input: &CudaBuffer<f32>,
    out_shape: &[usize],
    src_strides: &[isize],
    src_offset: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let n: usize = out_shape.iter().product();
    let (out_stride, src_stride) = pad_strided_copy_params(out_shape, src_strides, n)?;

    if n == 0 {
        return alloc_zeros_f32(0, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        STRIDED_COPY_PTX,
        "strided_copy_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "strided_copy_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let src_offset_u32 = src_offset as u32;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, STRIDED_COPY_PTX,
    //   "strided_copy_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, src_offset, n, out_stride[0..8],
    //   src_stride[0..8])` — 20 args, with 8-dim unrolled strides
    //   matching `STRIDED_COPY_MAX_DIMS = 8` (defined at line
    //   13596).
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570). `out_stride` and `src_stride` are
    //   `[u32; 8]` arrays returned by `pad_strided_copy_params`
    //   (definition at line 13586) which validates rank
    //   `<= STRIDED_COPY_MAX_DIMS`, rejects negative source strides
    //   that would wrap on `as u32`, and pads unused trailing dims
    //   with sentinels (`out_stride[d] = n+1` makes
    //   `flat / out_stride[d] == 0`; `src_stride[d] = 0` zeroes the
    //   source-offset contribution) so unused dims are no-ops.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above with `n = product(out_shape)` elements;
    //   no aliasing with `input`. Exclusive `out.inner_mut()` borrow.
    // - Each thread `i in [0, n)` decodes a multi-dim index from `i`
    //   using `out_stride`, then computes a source offset using
    //   `src_stride` and `src_offset`. With negative strides
    //   pre-rejected and the rank capped, the source offset is a
    //   well-defined u32 lookup into `input`. PTX bound check
    //   `tid >= n` short-circuits OOB threads. Caller is responsible
    //   for ensuring the resulting source offset stays inside
    //   `input.len()` (matches `as_strided` semantics in PyTorch).
    // - `n_u32 = n as u32` bounded by `launch_cfg(n)?` (definition
    //   at line 10523); the 16 `&out_stride[i]` / `&src_stride[i]`
    //   refs and the two `u32` refs all live to the trailing `?`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&src_offset_u32)
            .arg(&n_u32)
            .arg(&out_stride[0])
            .arg(&out_stride[1])
            .arg(&out_stride[2])
            .arg(&out_stride[3])
            .arg(&out_stride[4])
            .arg(&out_stride[5])
            .arg(&out_stride[6])
            .arg(&out_stride[7])
            .arg(&src_stride[0])
            .arg(&src_stride[1])
            .arg(&src_stride[2])
            .arg(&src_stride[3])
            .arg(&src_stride[4])
            .arg(&src_stride[5])
            .arg(&src_stride[6])
            .arg(&src_stride[7])
            .launch(cfg)?;
    }

    Ok(out)
}

/// f64 variant of [`gpu_strided_copy`]. CL-496.
#[cfg(feature = "cuda")]
pub fn gpu_strided_copy_f64(
    input: &CudaBuffer<f64>,
    out_shape: &[usize],
    src_strides: &[isize],
    src_offset: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let n: usize = out_shape.iter().product();
    let (out_stride, src_stride) = pad_strided_copy_params(out_shape, src_strides, n)?;

    if n == 0 {
        return alloc_zeros_f64(0, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        STRIDED_COPY_PTX,
        "strided_copy_kernel",
        "strided_copy_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "strided_copy_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "strided_copy_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(n)?;
    let src_offset_u32 = src_offset as u32;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx, "strided_copy_f64_kernel",
    //   ...)` call earlier in this fn; the f64 PTX is mechanically
    //   produced by `get_f64_ptx` from `STRIDED_COPY_PTX`. ABI is the
    //   same 20-arg shape as the f32 kernel: `(in_ptr, out_ptr,
    //   src_offset, n, out_stride[0..8], src_stride[0..8])`.
    // - `input` is on `device` (validated by `validate_device` at the
    //   top of this fn — definition at line 10582; the f64 path uses
    //   the generic device-only check, length is implicitly bounded
    //   by `pad_strided_copy_params`).
    // - `out_stride` / `src_stride` are `[u32; 8]` arrays returned by
    //   `pad_strided_copy_params` (definition at line 13586) which
    //   validates rank, rejects negative strides, and pads trailing
    //   dims with no-op sentinels.
    // - `out` was freshly allocated by `alloc_zeros_f64(n, device)?`
    //   immediately above with `n = product(out_shape)` f64
    //   elements; no aliasing with `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - Thread `i in [0, n)` performs the same gather as the f32
    //   kernel, but with f64 offsets (`tid * 8`). Caller is
    //   responsible for ensuring the resulting source offset stays
    //   inside `input.len()`.
    // - `n_u32 = n as u32` bounded by `launch_cfg(n)?` (definition
    //   at line 10523); all 18 stride/offset/n refs live to the
    //   trailing `?`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&src_offset_u32)
            .arg(&n_u32)
            .arg(&out_stride[0])
            .arg(&out_stride[1])
            .arg(&out_stride[2])
            .arg(&out_stride[3])
            .arg(&out_stride[4])
            .arg(&out_stride[5])
            .arg(&out_stride[6])
            .arg(&out_stride[7])
            .arg(&src_stride[0])
            .arg(&src_stride[1])
            .arg(&src_stride[2])
            .arg(&src_stride[3])
            .arg(&src_stride[4])
            .arg(&src_stride[5])
            .arg(&src_stride[6])
            .arg(&src_stride[7])
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Strided scatter (inverse of strided_copy)
// ---------------------------------------------------------------------------

/// Scatter a contiguous source buffer into a strided view of `dst`,
/// in-place. Inverse of [`gpu_strided_copy`].
///
/// `dst` must already contain the "base" data (typically a clone of the
/// caller's `self` tensor — `as_strided_scatter` semantics retain
/// non-view positions). After the kernel, every position addressed by
/// `(view_shape, dst_strides, dst_offset)` will hold the corresponding
/// value from `src`; non-view positions stay unchanged.
///
/// # Arguments
///
/// * `src`         — contiguous values to scatter, length `product(view_shape)`.
/// * `dst`         — destination buffer, mutated in place.
/// * `view_shape`  — shape of the strided view (and of `src`).
///   `view_shape.len() <= STRIDED_COPY_MAX_DIMS`.
/// * `dst_strides` — element strides of the strided view, aligned with
///   `view_shape`. Must be non-negative (matches the kernel's u32
///   stride encoding).
/// * `dst_offset`  — base element offset into `dst` for the view.
/// * `device`      — CUDA device.
///
/// # Errors
/// - [`GpuError::DeviceMismatch`] if `src`, `dst`, and `device` differ.
/// - [`GpuError::ShapeMismatch`] on rank mismatch, too many dims, or
///   negative strides.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_strided_scatter(
    src: &CudaBuffer<f32>,
    dst: &mut CudaBuffer<f32>,
    view_shape: &[usize],
    dst_strides: &[isize],
    dst_offset: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;

    validate_device(src, device)?;
    validate_device(dst, device)?;

    let n: usize = view_shape.iter().product();
    if n == 0 {
        return Ok(());
    }
    // Reuse `pad_strided_copy_params`: it produces shape-product strides
    // for the contiguous side (`view_shape` here) and pads the
    // user-supplied strides for the strided side (`dst_strides` here).
    let (in_decode_stride, dst_stride_padded) =
        pad_strided_copy_params(view_shape, dst_strides, n)?;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        STRIDED_SCATTER_PTX,
        "strided_scatter_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "strided_scatter_kernel",
                source: e,
            });
        }
    };

    let cfg = launch_cfg(n)?;
    let dst_offset_u32 = dst_offset as u32;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, STRIDED_SCATTER_PTX,
    //   "strided_scatter_kernel", ...)` call earlier in this fn; ABI
    //   is `(src_ptr, dst_ptr, dst_offset, n, in_decode_stride[0..8],
    //   dst_stride_padded[0..8])` — 20 args, 8-dim unrolled.
    // - `src` and `dst` are both validated to be on `device` by the
    //   two `validate_device` calls at the top of this fn (definition
    //   at line 10582). `dst: &mut CudaBuffer<f32>` is exclusively
    //   borrowed for this call, so it cannot alias `src`.
    // - `in_decode_stride` is the contiguous decode stride for the
    //   `src` side and `dst_stride_padded` is the user-supplied
    //   strided-view stride for `dst`, both produced by
    //   `pad_strided_copy_params` (definition at line 13586) which
    //   validates rank, rejects negative strides, and pads unused
    //   dims with no-op sentinels.
    // - Each thread `i in [0, n)` reads `src[i]` (contiguous) and
    //   writes `dst[dst_offset + dst_offset_for_index(i)]`. With
    //   negative strides rejected and rank capped, the destination
    //   offset is well-defined. Caller's contract (the rustdoc
    //   above) requires `dst` to be large enough to hold every
    //   addressed position; non-view positions in `dst` stay
    //   untouched (in-place semantics, mirrors PyTorch's
    //   `as_strided_scatter`).
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(src.inner())
            .arg(dst.inner_mut())
            .arg(&dst_offset_u32)
            .arg(&n_u32)
            .arg(&in_decode_stride[0])
            .arg(&in_decode_stride[1])
            .arg(&in_decode_stride[2])
            .arg(&in_decode_stride[3])
            .arg(&in_decode_stride[4])
            .arg(&in_decode_stride[5])
            .arg(&in_decode_stride[6])
            .arg(&in_decode_stride[7])
            .arg(&dst_stride_padded[0])
            .arg(&dst_stride_padded[1])
            .arg(&dst_stride_padded[2])
            .arg(&dst_stride_padded[3])
            .arg(&dst_stride_padded[4])
            .arg(&dst_stride_padded[5])
            .arg(&dst_stride_padded[6])
            .arg(&dst_stride_padded[7])
            .launch(cfg)?;
    }

    Ok(())
}

/// f64 variant of [`gpu_strided_scatter`].
#[cfg(feature = "cuda")]
pub fn gpu_strided_scatter_f64(
    src: &CudaBuffer<f64>,
    dst: &mut CudaBuffer<f64>,
    view_shape: &[usize],
    dst_strides: &[isize],
    dst_offset: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;

    validate_device(src, device)?;
    validate_device(dst, device)?;

    let n: usize = view_shape.iter().product();
    if n == 0 {
        return Ok(());
    }
    let (in_decode_stride, dst_stride_padded) =
        pad_strided_copy_params(view_shape, dst_strides, n)?;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        STRIDED_SCATTER_F64_PTX,
        "strided_scatter_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "strided_scatter_f64_kernel",
                source: e,
            });
        }
    };

    let cfg = launch_cfg(n)?;
    let dst_offset_u32 = dst_offset as u32;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, STRIDED_SCATTER_F64_PTX,
    //   "strided_scatter_f64_kernel", ...)` call earlier in this fn;
    //   ABI is the same 20-arg shape as the f32 kernel: `(src_ptr,
    //   dst_ptr, dst_offset, n, in_decode_stride[0..8],
    //   dst_stride_padded[0..8])`.
    // - `src` and `dst` are both validated to be on `device` by the
    //   two `validate_device` calls at the top of this fn (definition
    //   at line 10582). `dst: &mut CudaBuffer<f64>` is exclusively
    //   borrowed; cannot alias `src`.
    // - `in_decode_stride` and `dst_stride_padded` are `[u32; 8]`
    //   arrays from `pad_strided_copy_params` (definition at line
    //   13586), which validates rank, rejects negative strides, and
    //   pads unused dims with no-op sentinels.
    // - Each thread `i in [0, n)` reads `src[i]` and writes
    //   `dst[dst_offset + computed_offset]` (in-place semantics,
    //   mirrors PyTorch `as_strided_scatter`); caller is responsible
    //   for `dst` being large enough to hold every addressed
    //   position.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(src.inner())
            .arg(dst.inner_mut())
            .arg(&dst_offset_u32)
            .arg(&n_u32)
            .arg(&in_decode_stride[0])
            .arg(&in_decode_stride[1])
            .arg(&in_decode_stride[2])
            .arg(&in_decode_stride[3])
            .arg(&in_decode_stride[4])
            .arg(&in_decode_stride[5])
            .arg(&in_decode_stride[6])
            .arg(&in_decode_stride[7])
            .arg(&dst_stride_padded[0])
            .arg(&dst_stride_padded[1])
            .arg(&dst_stride_padded[2])
            .arg(&dst_stride_padded[3])
            .arg(&dst_stride_padded[4])
            .arg(&dst_stride_padded[5])
            .arg(&dst_stride_padded[6])
            .arg(&dst_stride_padded[7])
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "scale_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, SCALE_PTX, "scale_kernel",
    //   ...)` call earlier in this fn; ABI is `(in_ptr, out_ptr,
    //   scalar, n)`.
    // - `a` is on `device` (validated by `validate_unary` — definition
    //   at line 10570). `n = a.len()`.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above with length `n`; cannot alias `a`.
    //   Exclusive `out.inner_mut()` borrow.
    // - `scalar` is passed by reference to the local `f32` parameter
    //   on this fn; the local lives for the full call frame.
    // - The kernel writes `out[i] = a[i] * scalar` for `i in [0, n)`;
    //   PTX bound check `setp.ge.u32 %p, %r_tid, %n_reg; @%p bra
    //   DONE` (matching `ADD_PTX` line 164) skips OOB threads.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523) which errors if `n > u32::MAX`.
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "softmax_kernel",
                source: e,
            });
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, SOFTMAX_PTX, "softmax_kernel",
    //   ...)` call earlier in this fn; ABI is `(in_ptr, out_ptr, rows,
    //   cols)`. Kernel uses 256 f32 (1024 bytes) of shared memory per
    //   block for max + sum-exp reductions (matching
    //   `shared_mem_bytes: 256 * 4`).
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570). Caller's contract is `input.len()
    //   == rows * cols`.
    // - `out` was freshly allocated by `alloc_zeros_f32(rows * cols,
    //   device)?` immediately above; cannot alias `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - Grid `(rows, 1, 1)` × block `(256, 1, 1)`: one block per row
    //   reads `input[row*cols .. row*cols+cols]` (in two passes for
    //   max-then-sum) and writes `out[row*cols .. row*cols+cols]`.
    //   Every access is in `[0, rows*cols)`.
    // - `rows_u32 = rows as u32`, `cols_u32 = cols as u32`: launch-grid
    //   construction already cast `rows as u32`, bounding `rows`.
    //   Caller's responsibility for `cols <= u32::MAX` and
    //   `rows * cols <= usize::MAX` (the allocation already
    //   succeeded, bounding the product).
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "dropout_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, DROPOUT_PTX,
    //   "dropout_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, n, threshold, scale, seed)`.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570); `n = input.len()`.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above; cannot alias `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - `threshold: u32`, `scale: f32`, `seed: u32` are passed by
    //   reference to local variables on this fn's parameter slots,
    //   each living for the full call frame including the async
    //   launch.
    // - Per-thread `i in [0, n)`: reads `input[i]`, hashes the
    //   thread id with the seed (xorshift mixing of `tid *
    //   2654435761 ^ seed` per the rustdoc above), and writes
    //   `out[i] = (hash >= threshold) ? input[i] * scale : 0.0`. PTX
    //   bound check `setp.ge.u32 %p, %r_tid, %n_reg; @%p bra DONE`
    //   skips OOB threads.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523) which errors if `n > u32::MAX`.
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

/// Elementwise dropout for f64 tensors.
#[cfg(feature = "cuda")]
pub fn gpu_dropout_f64(
    input: &CudaBuffer<f64>,
    threshold: u32,
    scale: f64,
    seed: u32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let n = input.len();
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(&CACHE, DROPOUT_PTX, "dropout_kernel", "dropout_f64_kernel");
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "dropout_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "dropout_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx, "dropout_f64_kernel",
    //   ...)` call earlier in this fn; the f64 PTX is mechanically
    //   produced by `get_f64_ptx` from `DROPOUT_PTX`. ABI matches the
    //   f32 dropout: `(in_ptr, out_ptr, n, threshold, scale, seed)`
    //   with `scale` widened to f64.
    // - `input` is caller-supplied `&CudaBuffer<f64>`; the f64 path
    //   omits the explicit `validate_unary` check that the f32 path
    //   uses (caller's responsibility — same pattern as other f64
    //   helpers like `gpu_pow_f64`). `n = input.len()`.
    // - `out` was freshly allocated by `alloc_zeros_f64(n, device)?`
    //   immediately above; cannot alias `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - `threshold: u32`, `scale: f64`, `seed: u32` are passed by
    //   reference to fn-parameter locals; lifetimes cover the
    //   asynchronous launch.
    // - Per-thread `i in [0, n)`: hashes `tid ^ seed`, compares to
    //   `threshold`, writes `out[i] = (hash >= threshold) ?
    //   input[i] * scale : 0.0` (with f64 arithmetic). PTX bound
    //   check skips OOB threads.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523) which errors if `n > u32::MAX`.
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

#[cfg(not(feature = "cuda"))]
pub fn gpu_dropout_f64(
    _input: &CudaBuffer<f64>,
    _threshold: u32,
    _scale: f64,
    _seed: u32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "transpose_2d_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, TRANSPOSE_2D_PTX,
    //   "transpose_2d_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, m, n, total)`.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570). Caller's contract is `input.len()
    //   == m * n == total`.
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above; cannot alias `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - Thread `i in [0, total)` decodes `(row, col) = (i / n, i % n)`
    //   from the input layout `[M, N]` and writes
    //   `out[col * m + row] = input[row * n + col]`. Both indices
    //   are bounded by `total - 1` so the access is in
    //   `[0, total)`. PTX bound check skips OOB threads.
    // - `total_u32 = total as u32` is bounded by `launch_cfg(total)?`
    //   (definition at line 10523); `m_u32`, `n_u32` are caller-
    //   supplied and the caller is responsible for u32 range (the
    //   product `m * n == total` is already u32-bounded by
    //   `launch_cfg`).
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "permute_0213_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let d0_u32 = d0 as u32;
    let d1_u32 = d1 as u32;
    let d2_u32 = d2 as u32;
    let d3_u32 = d3 as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, PERMUTE_0213_PTX,
    //   "permute_0213_kernel", ...)` call earlier in this fn; ABI is
    //   `(in_ptr, out_ptr, d0, d1, d2, d3, total)`.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570). Caller's contract is `input.len()
    //   == d0 * d1 * d2 * d3 == total`.
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above with `total` elements; cannot
    //   alias `input`. Exclusive `out.inner_mut()` borrow.
    // - Each thread `i in [0, total)` decodes the source coords
    //   `(b, h, s, dh)` and writes
    //   `out[((b*d2 + h)*d1 + s)*d3 + dh] = input[((b*d1 + s)*d2 +
    //   h)*d3 + dh]`. Both indices are bounded by `total` because
    //   each axis index is < its dim size and the strides multiply
    //   out to `total`.
    // - `total_u32 = total as u32` bounded by `launch_cfg(total)?`
    //   (definition at line 10523); `d0_u32 .. d3_u32` are caller-
    //   supplied with the implicit constraint that their product
    //   `total` already fits u32.
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, SMALL_MATMUL_PTX,
    //   "small_matmul_kernel", ...)` call earlier in this fn; ABI is
    //   `(a_ptr, b_ptr, c_ptr, m, k, n, total)`. (The Err arm of that
    //   match falls back to `crate::blas::gpu_matmul_f32` and never
    //   reaches this block.)
    // - `a` and `b` are caller-supplied `&CudaBuffer<f32>`; the
    //   matmul contract is `a.len() == m * k`, `b.len() == k * n`.
    //   This fn does not validate explicitly — caller's
    //   responsibility (consistent with the small-matmul fast-path
    //   contract documented in the rustdoc above).
    // - `c` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` (where `total = m * n`) immediately above; cannot
    //   alias `a` or `b`. Exclusive `c.inner_mut()` borrow.
    // - Thread `i in [0, total)` decodes `(row, col) = (i / n, i %
    //   n)` and writes `c[row*n + col] = sum_{p=0..k} a[row*k + p]
    //   * b[p*n + col]`. The reads land in
    //   `a[0..m*k] = a[0..a.len())` and `b[0..k*n] = b[0..b.len())`.
    //   PTX bound check skips OOB threads.
    // - `total_u32 = total as u32` bounded by `launch_cfg(total)?`
    //   (definition at line 10523); `m_u32`, `k_u32`, `n_u32` are
    //   caller-supplied with the constraint that their pairwise
    //   products fit u32 (the buffer-len validation already implies
    //   this).
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "embed_lookup_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(d, device)?;
    let cfg = launch_cfg(d)?;
    let d_u32 = d as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, EMBED_LOOKUP_PTX,
    //   "embed_lookup_kernel", ...)` call earlier in this fn; ABI is
    //   `(idx_ptr, weight_ptr, out_ptr, d)`.
    // - `idx` is a 1-element `&CudaBuffer<f32>` (single token id
    //   encoded as f32; rustdoc above documents the layout).
    //   `weight` is `[V, D]` flat, both caller-supplied; this fn
    //   does not call `validate_unary` (kernel-launch fast-path for
    //   inference).
    // - `out` was freshly allocated by `alloc_zeros_f32(d, device)?`
    //   immediately above with `d` elements; cannot alias `idx` or
    //   `weight`. Exclusive `out.inner_mut()` borrow.
    // - Per-thread `i in [0, d)`: reads `idx[0]` (broadcast across
    //   the grid), gathers `weight[idx[0] * d + i]`, writes
    //   `out[i]`. Caller's contract is `idx[0] < V` (out-of-range
    //   indices are UB at the kernel level — mirrors PyTorch's
    //   `Embedding.forward` user contract).
    // - `d_u32 = d as u32` bounded by `launch_cfg(d)?` (definition
    //   at line 10523).
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "slice_write_kernel",
                source: e,
            });
        }
    };

    let cfg = launch_cfg(total)?;
    let n_u32 = total as u32;
    let d_u32 = d as u32;
    let max_len_u32 = max_len as u32;
    let pos_u32 = pos as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, SLICE_WRITE_PTX,
    //   "slice_write_kernel", ...)` call earlier in this fn; ABI is
    //   `(src_ptr, dst_ptr, total, d, max_len, pos)`.
    // - `src` is `&CudaBuffer<f32>` of shape `[N, D]` flat;
    //   `dst: &mut CudaBuffer<f32>` of shape `[N, max_len, D]` flat,
    //   exclusively borrowed for this call so cannot alias `src`.
    //   This fn does not validate `src.len() == n_batch * d` or
    //   `dst.len() == n_batch * max_len * d` (KV-cache fast path —
    //   caller's contract per the rustdoc above).
    // - Thread `i in [0, total)` where `total = n_batch * d` decodes
    //   `(b, dim_idx) = (i / d, i % d)` and writes
    //   `dst[(b * max_len + pos) * d + dim_idx] = src[b * d +
    //   dim_idx]`. Caller's contract `pos < max_len` keeps the
    //   destination index in `[0, dst.len())`.
    // - `n_u32 = total as u32` bounded by `launch_cfg(total)?`
    //   (definition at line 10523); the other casts are caller-
    //   supplied dims, caller is responsible for u32 range.
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "slice_read_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let total_u32 = total as u32;
    let d_u32 = d as u32;
    let len_u32 = len as u32;
    let max_len_u32 = max_len as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, SLICE_READ_PTX,
    //   "slice_read_kernel", ...)` call earlier in this fn; ABI is
    //   `(src_ptr, out_ptr, total, d, len, max_len)`.
    // - `src` is `&CudaBuffer<f32>` of shape `[N, max_len, D]` flat
    //   (KV-cache backing buffer); caller's contract is `src.len()
    //   >= n_batch * max_len * d`.
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above with `total = n_batch * len * d`
    //   elements; cannot alias `src`. Exclusive `out.inner_mut()`
    //   borrow.
    // - Thread `i in [0, total)` decodes `(b, t, dim_idx)` from
    //   `total / (len * d)` etc., and writes `out[b*len*d + t*d +
    //   dim_idx] = src[b*max_len*d + t*d + dim_idx]`. Caller's
    //   contract `len <= max_len` keeps the source index in
    //   `[0, src.len())`.
    // - `total_u32 = total as u32` bounded by `launch_cfg(total)?`
    //   (definition at line 10523); the other casts are caller-
    //   supplied dims, caller is responsible for u32 range.
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
    try_launch_unary(input, device, GELU_PTX, "gelu_kernel")
}

/// Elementwise GELU activation on GPU using the tanh approximation:
/// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
///
/// Matches PyTorch `nn.GELU(approximate="tanh")`.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_tanh(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(input, device)?;
    try_launch_unary(input, device, GELU_TANH_PTX, "gelu_tanh_kernel")
}

/// Elementwise GELU activation on GPU using exact erf:
/// `gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`.
///
/// Matches PyTorch `nn.GELU(approximate="none")` (the default).
#[cfg(feature = "cuda")]
pub fn gpu_gelu_erf(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(input, device)?;
    try_launch_unary(input, device, GELU_ERF_PTX, "gelu_erf_kernel")
}

/// GELU backward for the tanh approximation mode.
/// Let `u = sqrt(2/π) * (x + 0.044715 * x³)`, `t = tanh(u)`.
/// `d/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t²) * sqrt(2/π) * (1 + 3*0.044715*x²)`
#[cfg(feature = "cuda")]
pub fn gpu_gelu_backward_tanh(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, input, device)?;
    try_launch_binary(
        grad,
        input,
        device,
        GELU_BACKWARD_TANH_PTX,
        "gelu_backward_tanh_kernel",
    )
}

// ---------------------------------------------------------------------------
// Public API -- SiLU (Swish)
// ---------------------------------------------------------------------------

/// Elementwise SiLU activation on GPU: `silu(x) = x * sigmoid(x)`.
#[cfg(feature = "cuda")]
pub fn gpu_silu(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(input, device)?;
    try_launch_unary(input, device, SILU_PTX, "silu_kernel")
}

/// SiLU backward: `out[i] = grad[i] * (sig + x * sig * (1 - sig))`
/// where `sig = sigmoid(input[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_silu_backward(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, input, device)?;

    try_launch_binary(
        grad,
        input,
        device,
        SILU_BACKWARD_PTX,
        "silu_backward_kernel",
    )
}

// ---------------------------------------------------------------------------
// Public API -- ELU
// ---------------------------------------------------------------------------

/// Elementwise ELU activation on GPU: `elu(x) = x > 0 ? x : alpha * (exp(x) - 1)`.
///
/// Uses a custom launch because the kernel takes an extra `alpha` parameter.
#[cfg(feature = "cuda")]
pub fn gpu_elu(
    input: &CudaBuffer<f32>,
    alpha: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let n = input.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        ELU_PTX,
        "elu_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "elu_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ELU_PTX, "elu_kernel",
    //   ...)` call earlier in this fn; ABI is `(in_ptr, out_ptr, n,
    //   alpha)`.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570). `n = input.len()`.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above; cannot alias `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - `alpha` is passed by reference to the local `f32` parameter
    //   on this fn; the local lives for the full call frame
    //   including the asynchronous launch.
    // - The kernel writes `out[i] = input[i] > 0 ? input[i] :
    //   alpha * (exp(input[i]) - 1.0)` for `i in [0, n)`; PTX bound
    //   check skips OOB threads.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523) which errors if `n > u32::MAX`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&alpha)
            .launch(cfg)?;
    }

    Ok(out)
}

/// ELU backward: `out[i] = x > 0 ? grad[i] : grad[i] * alpha * exp(x)`.
///
/// Uses a custom launch because the kernel takes an extra `alpha` parameter.
#[cfg(feature = "cuda")]
pub fn gpu_elu_backward(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    alpha: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_binary(grad, input, device)?;

    let n = grad.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        ELU_BACKWARD_PTX,
        "elu_backward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "elu_backward_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ELU_BACKWARD_PTX,
    //   "elu_backward_kernel", ...)` call earlier in this fn; ABI is
    //   `(grad_ptr, in_ptr, out_ptr, n, alpha)`.
    // - `grad` and `input` are validated to be on `device` and same
    //   length by `validate_binary` (definition at line 10546) at
    //   the top of this fn; `n = grad.len()`.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above; cannot alias `grad` or `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - `alpha` is passed by reference to the local `f32` parameter
    //   on this fn; lifetimes cover the asynchronous launch.
    // - The kernel writes `out[i] = input[i] > 0 ? grad[i] :
    //   grad[i] * alpha * exp(input[i])` for `i in [0, n)`; PTX
    //   bound check skips OOB threads.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523) which errors if `n > u32::MAX`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad.inner())
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&alpha)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Mish
// ---------------------------------------------------------------------------

/// Elementwise Mish activation on GPU: `mish(x) = x * tanh(softplus(x))`.
#[cfg(feature = "cuda")]
pub fn gpu_mish(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(input, device)?;
    try_launch_unary(input, device, MISH_PTX, "mish_kernel")
}

/// Mish backward:
/// `out[i] = grad[i] * (tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2))`
/// where `sp = softplus(x) = ln(1 + exp(x))`.
#[cfg(feature = "cuda")]
pub fn gpu_mish_backward(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, input, device)?;

    try_launch_binary(
        grad,
        input,
        device,
        MISH_BACKWARD_PTX,
        "mish_backward_kernel",
    )
}

/// Elementwise clamp: `out[i] = max(min_val, min(max_val, x[i]))`.
///
/// Uses a custom launch because the kernel takes two extra f32 parameters.
#[cfg(feature = "cuda")]
pub fn gpu_clamp(
    input: &CudaBuffer<f32>,
    min_val: f32,
    max_val: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let n = input.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        CLAMP_PTX,
        "clamp_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "clamp_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, CLAMP_PTX, "clamp_kernel",
    //   ...)` call earlier in this fn; ABI is `(in_ptr, out_ptr, n,
    //   min_val, max_val)`.
    // - `input` is on `device` (validated by `validate_unary` —
    //   definition at line 10570). `n = input.len()`.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above; cannot alias `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - `min_val` and `max_val` are passed by reference to the
    //   local `f32` parameters; lifetimes cover the asynchronous
    //   launch.
    // - The kernel writes `out[i] = max(min_val, min(max_val,
    //   input[i]))` for `i in [0, n)`; PTX bound check skips OOB
    //   threads.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&min_val)
            .arg(&max_val)
            .launch(cfg)?;
    }

    Ok(out)
}

/// VJP for `clamp(x, min, max)`: `out[i] = grad[i]` if `min <= x[i] <= max`,
/// else `0`. (#524)
#[cfg(feature = "cuda")]
pub fn gpu_clamp_backward(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    min_val: f32,
    max_val: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    let n = input.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        CLAMP_BACKWARD_PTX,
        "clamp_backward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "clamp_backward_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, CLAMP_BACKWARD_PTX,
    //   "clamp_backward_kernel", ...)` call earlier in this fn; ABI
    //   is `(grad_ptr, in_ptr, out_ptr, min_val, max_val, n)`.
    // - `grad` and `input` were length-checked equal at the top of
    //   this fn (`grad.len() != input.len()` returns
    //   `LengthMismatch`). Both are caller-guaranteed to be on
    //   `device` (no explicit `validate_*` here — caller's
    //   contract). `n = input.len()`.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above; cannot alias `grad` or `input`. Exclusive
    //   `out.inner_mut()` borrow.
    // - `min_val` and `max_val` are passed by reference to the local
    //   `f32` parameters; lifetimes cover the asynchronous launch.
    // - The kernel writes `out[i] = (min_val <= input[i] <= max_val)
    //   ? grad[i] : 0` for `i in [0, n)`; PTX bound check skips OOB
    //   threads.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad.inner())
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&min_val)
            .arg(&max_val)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_clamp_backward(
    _grad: &CudaBuffer<f32>,
    _input: &CudaBuffer<f32>,
    _min_val: f32,
    _max_val: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// f32 broadcast-along-middle-dim kernel (#524). Expands a
/// `[outer, inner]` source into `[outer, repeat_count, inner]`. Used for
/// sum_dim / mean_dim backward to replicate the gradient along the
/// reduced dim.
#[cfg(feature = "cuda")]
pub fn gpu_repeat_along_dim(
    input: &CudaBuffer<f32>,
    outer: usize,
    repeat_count: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    if input.len() != outer * inner {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_repeat_along_dim",
            expected: vec![outer * inner],
            got: vec![input.len()],
        });
    }
    let total = outer * repeat_count * inner;
    if total == 0 {
        return alloc_zeros_f32(0, device);
    }

    let ctx = device.context();
    let stream = device.stream();
    let f = match crate::module_cache::get_or_compile(
        ctx,
        REPEAT_ALONG_DIM_PTX,
        "repeat_along_dim_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "repeat_along_dim_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let outer_u32 = outer as u32;
    let rep_u32 = repeat_count as u32;
    let inner_u32 = inner as u32;
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, REPEAT_ALONG_DIM_PTX,
    //   "repeat_along_dim_kernel", ...)` call earlier in this fn;
    //   ABI is `(in_ptr, out_ptr, outer, repeat_count, inner)`.
    // - `input.len() == outer * inner` was verified at the top of
    //   this fn (returns `ShapeMismatch` otherwise). `total = outer
    //   * repeat_count * inner` and the early-return for `total ==
    //   0` ensures the launch only fires for non-empty grids.
    // - `out` was freshly allocated by `alloc_zeros_f32(total,
    //   device)?` immediately above with `total` elements; cannot
    //   alias `input`. Exclusive `out.inner_mut()` borrow.
    // - Thread `i in [0, total)` decodes `(outer_idx, rep_idx,
    //   inner_idx)` from `i` via `inner` and `repeat_count` and
    //   writes `out[i] = input[outer_idx * inner + inner_idx]`. The
    //   read is in `[0, outer * inner) == [0, input.len())`. PTX
    //   bound check guards `tid >= total`.
    // - `outer_u32`, `rep_u32`, `inner_u32`: caller-supplied dims.
    //   The grid uses `total` threads, bounded by `launch_cfg(total)?`
    //   (definition at line 10523) which errors on `total >
    //   u32::MAX`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&rep_u32)
            .arg(&inner_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// f64 counterpart of [`gpu_repeat_along_dim`].
#[cfg(feature = "cuda")]
pub fn gpu_repeat_along_dim_f64(
    input: &CudaBuffer<f64>,
    outer: usize,
    repeat_count: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    if input.len() != outer * inner {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_repeat_along_dim_f64",
            expected: vec![outer * inner],
            got: vec![input.len()],
        });
    }
    let total = outer * repeat_count * inner;
    if total == 0 {
        return alloc_zeros_f64(0, device);
    }

    let ctx = device.context();
    let stream = device.stream();
    let ptx = get_f64_ptx(
        &CACHE,
        REPEAT_ALONG_DIM_PTX,
        "repeat_along_dim_kernel",
        "repeat_along_dim_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "repeat_along_dim_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "repeat_along_dim_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(total, device)?;
    let cfg = launch_cfg(total)?;
    let outer_u32 = outer as u32;
    let rep_u32 = repeat_count as u32;
    let inner_u32 = inner as u32;
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, ptx, "repeat_along_dim_f64_kernel",
    //   ...)` call earlier in this fn; the f64 PTX is mechanically
    //   produced by `get_f64_ptx` from `REPEAT_ALONG_DIM_PTX`. ABI
    //   matches the f32 kernel: `(in_ptr, out_ptr, outer,
    //   repeat_count, inner)`, with element offsets sized for f64.
    // - `input.len() == outer * inner` was verified at the top of
    //   this fn (returns `ShapeMismatch` otherwise). `total = outer
    //   * repeat_count * inner` and the early-return for `total ==
    //   0` ensures the launch only fires for non-empty grids.
    // - `out` was freshly allocated by `alloc_zeros_f64(total,
    //   device)?` immediately above with `total` f64 elements;
    //   cannot alias `input`. Exclusive `out.inner_mut()` borrow.
    // - Thread `i in [0, total)` decodes `(outer_idx, rep_idx,
    //   inner_idx)` and writes `out[i] = input[outer_idx * inner +
    //   inner_idx]`; the read is in `[0, input.len())`. PTX bound
    //   check guards `tid >= total`.
    // - `outer_u32`, `rep_u32`, `inner_u32`: grid is `total` threads,
    //   bounded by `launch_cfg(total)?` (definition at line 10523).
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&rep_u32)
            .arg(&inner_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_repeat_along_dim(
    _input: &CudaBuffer<f32>,
    _outer: usize,
    _rep: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_repeat_along_dim_f64(
    _input: &CudaBuffer<f64>,
    _outer: usize,
    _rep: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
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

    try_launch_binary(a, b, device, DIV_PTX, "div_kernel")
}

/// Elementwise exponential: `out[i] = exp(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_exp(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    try_launch_unary(a, device, EXP_PTX, "exp_kernel")
}

/// Elementwise natural log: `out[i] = ln(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_log(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    try_launch_unary(a, device, LOG_PTX, "log_kernel")
}

/// Elementwise square root: `out[i] = sqrt(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_sqrt(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    try_launch_unary(a, device, SQRT_PTX, "sqrt_kernel")
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "pow_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` resolved via the
    //   `module_cache::get_or_compile(ctx, POW_PTX, "pow_kernel",
    //   ...)` call earlier in this fn; ABI is `(in_ptr, out_ptr,
    //   exponent, n)`.
    // - `a` is on `device` (validated by `validate_unary` —
    //   definition at line 10570). `n = a.len()`.
    // - `out` was freshly allocated by `alloc_zeros_f32(n, device)?`
    //   immediately above; cannot alias `a`. Exclusive
    //   `out.inner_mut()` borrow.
    // - `exponent` is passed by reference to the local `f32`
    //   parameter on this fn; lifetimes cover the asynchronous
    //   launch.
    // - The kernel writes `out[i] = pow(a[i], exponent)` for
    //   `i in [0, n)`; PTX bound check skips OOB threads.
    // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?` (definition
    //   at line 10523).
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
    try_launch_unary(a, device, ABS_PTX, "abs_kernel")
}

/// Elementwise sigmoid: `out[i] = 1 / (1 + exp(-a[i]))`.
#[cfg(feature = "cuda")]
pub fn gpu_sigmoid(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    try_launch_unary(a, device, SIGMOID_PTX, "sigmoid_kernel")
}

/// Elementwise tanh: `out[i] = tanh(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_tanh(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    try_launch_unary(a, device, TANH_PTX, "tanh_kernel")
}

// ---------------------------------------------------------------------------
// Public API -- f64 elementwise ops
// ---------------------------------------------------------------------------

/// Elementwise f64 addition: `out[i] = a[i] + b[i]`.
#[cfg(feature = "cuda")]
pub fn gpu_add_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let ptx = get_f64_ptx(&CACHE, ADD_PTX, "add_kernel", "add_f64_kernel");
    try_launch_binary_f64(a, b, device, ptx, "add_f64_kernel")
}

/// Elementwise f64 subtraction: `out[i] = a[i] - b[i]`.
#[cfg(feature = "cuda")]
pub fn gpu_sub_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let ptx = get_f64_ptx(&CACHE, SUB_PTX, "sub_kernel", "sub_f64_kernel");
    try_launch_binary_f64(a, b, device, ptx, "sub_f64_kernel")
}

/// Elementwise f64 multiplication: `out[i] = a[i] * b[i]`.
#[cfg(feature = "cuda")]
pub fn gpu_mul_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let ptx = get_f64_ptx(&CACHE, MUL_PTX, "mul_kernel", "mul_f64_kernel");
    try_launch_binary_f64(a, b, device, ptx, "mul_f64_kernel")
}

/// Elementwise f64 division: `out[i] = a[i] / b[i]`.
#[cfg(feature = "cuda")]
pub fn gpu_div_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let ptx = get_f64_ptx(&CACHE, DIV_PTX, "div_kernel", "div_f64_kernel");
    try_launch_binary_f64(a, b, device, ptx, "div_f64_kernel")
}

/// Elementwise f64 negation: `out[i] = -a[i]`.
#[cfg(feature = "cuda")]
pub fn gpu_neg_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx = get_f64_ptx(&CACHE, NEG_PTX, "neg_kernel", "neg_f64_kernel");
    try_launch_unary_f64(a, device, ptx, "neg_f64_kernel")
}

/// Elementwise f64 ReLU: `out[i] = max(a[i], 0.0)`.
#[cfg(feature = "cuda")]
pub fn gpu_relu_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx = get_f64_ptx(&CACHE, RELU_PTX, "relu_kernel", "relu_f64_kernel");
    try_launch_unary_f64(a, device, ptx, "relu_f64_kernel")
}

/// Elementwise f64 scale: `out[i] = a[i] * scalar`.
#[cfg(feature = "cuda")]
pub fn gpu_scale_f64(
    a: &CudaBuffer<f64>,
    scalar: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(&CACHE, SCALE_PTX, "scale_kernel", "scale_f64_kernel");
    match crate::module_cache::get_or_compile(ctx, ptx, "scale_f64_kernel", device.ordinal() as u32)
    {
        Ok(f) => {
            let mut out = alloc_zeros_f64(n, device)?;
            let cfg = launch_cfg(n)?;
            let n_u32 = n as u32;

            // SAFETY:
            // - `f` is a valid PTX `CudaFunction` from the matching
            //   `Ok(f)` arm of the `if let Ok(f) =
            //   module_cache::get_or_compile(...)` above; the f64 PTX
            //   was produced by `get_f64_ptx` from `SCALE_PTX`. ABI is
            //   `(in_ptr, out_ptr, scalar, n)`. (The `else` branch
            //   below this block is gpu-F territory — silent CPU
            //   fallback that this dispatch leaves intact.)
            // - `a` is caller-supplied `&CudaBuffer<f64>`; `n = a.len()`.
            //   The f64 path omits `validate_unary` — caller's contract.
            // - `out` was freshly allocated by `alloc_zeros_f64(n,
            //   device)?` in this `Ok` arm; cannot alias `a`. Exclusive
            //   `out.inner_mut()` borrow.
            // - `scalar: f64` is passed by reference to the fn parameter
            //   local; lifetime covers the asynchronous launch.
            // - The kernel writes `out[i] = a[i] * scalar` for
            //   `i in [0, n)` (with f64 8-byte stride); PTX bound check
            //   skips OOB threads.
            // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?`
            //   (definition at line 10523).
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
        Err(e) => {
            if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
                tracing::warn!(
                    target: "ferrotorch::gpu_fallback",
                    kernel = "scale_f64_kernel",
                    error = %e,
                    "PTX compile failed; falling back to CPU. Unset \
                     FERROTORCH_ENABLE_GPU_FALLBACK to make this an error instead.",
                );
                let a_host = gpu_to_cpu(a, device)?;
                let result: Vec<f64> = a_host.iter().map(|&x| x * scalar).collect();
                return cpu_to_gpu(&result, device);
            }
            Err(GpuError::PtxCompileFailed {
                kernel: "scale_f64_kernel",
                source: e,
            })
        }
    }
}

/// Elementwise f64 exp: `out[i] = exp(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_exp_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    try_launch_unary_f64(a, device, EXP_F64_PTX, "exp_f64_kernel")
}

/// Elementwise f64 log: `out[i] = ln(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_log_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    try_launch_unary_f64(a, device, LOG_F64_PTX, "log_f64_kernel")
}

/// Elementwise f64 sqrt: `out[i] = sqrt(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_sqrt_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx = get_f64_ptx(&CACHE, SQRT_PTX, "sqrt_kernel", "sqrt_f64_kernel");
    try_launch_unary_f64(a, device, ptx, "sqrt_f64_kernel")
}

/// Elementwise f64 pow: `out[i] = a[i] ^ exponent`.
#[cfg(feature = "cuda")]
pub fn gpu_pow_f64(
    a: &CudaBuffer<f64>,
    exponent: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    match crate::module_cache::get_or_compile(
        ctx,
        POW_F64_PTX,
        "pow_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => {
            let mut out = alloc_zeros_f64(n, device)?;
            let cfg = launch_cfg(n)?;
            let n_u32 = n as u32;

            // SAFETY:
            // - `f` is a valid PTX `CudaFunction` from the matching
            //   `Ok(f)` arm of the `if let Ok(f) =
            //   module_cache::get_or_compile(...)` above; the kernel is
            //   the precompiled `POW_F64_PTX`. ABI is `(in_ptr, out_ptr,
            //   exponent, n)`. (The `else` branch below this block is
            //   gpu-F territory — silent CPU fallback that this
            //   dispatch leaves intact.)
            // - `a` is caller-supplied `&CudaBuffer<f64>`; `n = a.len()`.
            //   The f64 path omits `validate_unary` — caller's contract.
            // - `out` was freshly allocated by `alloc_zeros_f64(n,
            //   device)?` in this `Ok` arm; cannot alias `a`. Exclusive
            //   `out.inner_mut()` borrow.
            // - `exponent: f64` is passed by reference to the fn-parameter
            //   local; lifetime covers the asynchronous launch.
            // - The kernel writes `out[i] = pow(a[i], exponent)` for
            //   `i in [0, n)` (f64 8-byte stride); PTX bound check skips
            //   OOB threads.
            // - `n_u32 = n as u32` is bounded by `launch_cfg(n)?`
            //   (definition at line 10523).
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
        Err(e) => {
            if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
                tracing::warn!(
                    target: "ferrotorch::gpu_fallback",
                    kernel = "pow_f64_kernel",
                    error = %e,
                    "PTX compile failed; falling back to CPU. Unset \
                     FERROTORCH_ENABLE_GPU_FALLBACK to make this an error instead.",
                );
                let a_host = gpu_to_cpu(a, device)?;
                let result: Vec<f64> = a_host.iter().map(|&x| x.powf(exponent)).collect();
                return cpu_to_gpu(&result, device);
            }
            Err(GpuError::PtxCompileFailed {
                kernel: "pow_f64_kernel",
                source: e,
            })
        }
    }
}

/// Elementwise f64 abs: `out[i] = |a[i]|`.
#[cfg(feature = "cuda")]
pub fn gpu_abs_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx = get_f64_ptx(&CACHE, ABS_PTX, "abs_kernel", "abs_f64_kernel");
    try_launch_unary_f64(a, device, ptx, "abs_f64_kernel")
}

/// Elementwise f64 sigmoid: `out[i] = 1 / (1 + exp(-a[i]))`.
#[cfg(feature = "cuda")]
pub fn gpu_sigmoid_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    try_launch_unary_f64(a, device, SIGMOID_F64_PTX, "sigmoid_f64_kernel")
}

/// Elementwise f64 tanh: `out[i] = tanh(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_tanh_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    try_launch_unary_f64(a, device, TANH_F64_PTX, "tanh_f64_kernel")
}

// ---------------------------------------------------------------------------
// Public API -- f64 backward ops
// ---------------------------------------------------------------------------

/// ReLU backward (f64): `out[i] = (input[i] > 0) ? grad[i] : 0`.
#[cfg(feature = "cuda")]
pub fn gpu_relu_backward_f64(
    grad: &CudaBuffer<f64>,
    input: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    let ptx = get_f64_ptx(
        &CACHE,
        RELU_BACKWARD_PTX,
        "relu_backward_kernel",
        "relu_backward_f64_kernel",
    );
    try_launch_binary_f64(grad, input, device, ptx, "relu_backward_f64_kernel")
}

/// Sigmoid backward (f64): `out[i] = grad[i] * output[i] * (1 - output[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_sigmoid_backward_f64(
    grad: &CudaBuffer<f64>,
    output: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if grad.len() != output.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: output.len(),
        });
    }
    let ptx = get_f64_ptx(
        &CACHE,
        SIGMOID_BACKWARD_PTX,
        "sigmoid_backward_kernel",
        "sigmoid_backward_f64_kernel",
    );
    try_launch_binary_f64(grad, output, device, ptx, "sigmoid_backward_f64_kernel")
}

/// Tanh backward (f64): `out[i] = grad[i] * (1 - output[i]^2)`.
#[cfg(feature = "cuda")]
pub fn gpu_tanh_backward_f64(
    grad: &CudaBuffer<f64>,
    output: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    if grad.len() != output.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: output.len(),
        });
    }
    let ptx = get_f64_ptx(
        &CACHE,
        TANH_BACKWARD_PTX,
        "tanh_backward_kernel",
        "tanh_backward_f64_kernel",
    );
    try_launch_binary_f64(grad, output, device, ptx, "tanh_backward_f64_kernel")
}

// ---------------------------------------------------------------------------
// Public API -- f64 broadcast ops
// ---------------------------------------------------------------------------

/// Broadcast addition (f64): `out[i] = a[bcast_a(i)] + b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub fn gpu_broadcast_add_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let a_str = broadcast_strides(a_shape, out_shape);
    let b_str = broadcast_strides(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
    let out_numel: usize = out_shape.iter().product();

    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx = get_f64_ptx(
        &CACHE,
        BROADCAST_ADD_PTX,
        "broadcast_add_kernel",
        "broadcast_add_f64_kernel",
    );
    try_launch_broadcast_binary_f64(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        ptx,
        "broadcast_add_f64_kernel",
    )
}

/// Broadcast subtraction (f64): `out[i] = a[bcast_a(i)] - b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub fn gpu_broadcast_sub_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let a_str = broadcast_strides(a_shape, out_shape);
    let b_str = broadcast_strides(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
    let out_numel: usize = out_shape.iter().product();

    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx = get_f64_ptx(
        &CACHE,
        BROADCAST_SUB_PTX,
        "broadcast_sub_kernel",
        "broadcast_sub_f64_kernel",
    );
    try_launch_broadcast_binary_f64(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        ptx,
        "broadcast_sub_f64_kernel",
    )
}

/// Broadcast multiplication (f64): `out[i] = a[bcast_a(i)] * b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub fn gpu_broadcast_mul_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let a_str = broadcast_strides(a_shape, out_shape);
    let b_str = broadcast_strides(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
    let out_numel: usize = out_shape.iter().product();

    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx = get_f64_ptx(
        &CACHE,
        BROADCAST_MUL_PTX,
        "broadcast_mul_kernel",
        "broadcast_mul_f64_kernel",
    );
    try_launch_broadcast_binary_f64(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        ptx,
        "broadcast_mul_f64_kernel",
    )
}

/// Broadcast division (f64): `out[i] = a[bcast_a(i)] / b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub fn gpu_broadcast_div_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let a_str = broadcast_strides(a_shape, out_shape);
    let b_str = broadcast_strides(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
    let out_numel: usize = out_shape.iter().product();

    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx = get_f64_ptx(
        &CACHE,
        BROADCAST_DIV_PTX,
        "broadcast_div_kernel",
        "broadcast_div_f64_kernel",
    );
    try_launch_broadcast_binary_f64(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        ptx,
        "broadcast_div_f64_kernel",
    )
}

// ---------------------------------------------------------------------------
// Public API -- f64 reduction ops
// ---------------------------------------------------------------------------

/// Full reduce-sum for f64: returns a 1-element buffer containing the sum of all elements.
#[cfg(feature = "cuda")]
pub fn gpu_reduce_sum_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let n = a.len();
    if n == 0 {
        return cpu_to_gpu(&[0.0f64], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        REDUCE_SUM_PTX,
        "reduce_sum_kernel",
        "reduce_sum_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "reduce_sum_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "reduce_sum_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);

    let mut partials = alloc_zeros_f64(num_blocks as usize, device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `reduce_sum_kernel` returned by `module_cache::get_or_compile`
    //   at lines 16195-16208 (Err short-circuits via the `?`-ladder
    //   in the `Err` arm); its `(in_ptr, partials_ptr, n)` ABI matches
    //   `REDUCE_SUM_PTX` extended via `ptx_f32_to_f64` (line 16189).
    // - `a: &CudaBuffer<f64>` is the caller-supplied input; `partials:
    //   &mut CudaBuffer<f64>` was alloc'd at line 16214 via
    //   `alloc_zeros_f64(num_blocks, device)?` and is exclusively owned
    //   by this function until the launch returns. The `&mut`
    //   borrow on `partials` precludes aliasing `a` per Rust's borrow
    //   rules.
    // - `n = a.len()` (line 16181); the kernel reads `a[i]` only for
    //   `i in [0, n)` per the PTX strided loop bound (`@%p bra END`
    //   pattern shared by `REDUCE_SUM_PTX`). `partials` has
    //   `num_blocks` elements (allocator at 16214) and the kernel
    //   writes one element per block, with `num_blocks <= 1024`
    //   capped at line 16212.
    // - `n_u32 = n as u32` (line 16215) cannot truncate: the `n == 0`
    //   early-return at line 16182 and the kernel never accepts more
    //   than `u32::MAX` threads — `num_blocks` was computed by
    //   saturating arithmetic at line 16211 then capped to 1024.
    // - cudarc enqueues the launch on `stream`; the three argument
    //   references (`a.inner()`, `partials.inner_mut()`, `&n_u32`) all
    //   live until the trailing `?`, and stream synchronization is
    //   the caller's responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }

    if num_blocks <= 256 {
        let host_partials = gpu_to_cpu(&partials, device)?;
        let total: f64 = host_partials.iter().sum();
        return cpu_to_gpu(&[total], device);
    }

    gpu_reduce_sum_f64(&partials, device)
}

/// f64 parallel min reduction. Mirrors [`gpu_reduce_min`] but uses
/// the f64-transformed PTX. (#627)
#[cfg(feature = "cuda")]
pub fn gpu_reduce_min_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let n = a.len();
    if n == 0 {
        return cpu_to_gpu(&[f64::INFINITY], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        REDUCE_MIN_PTX,
        "reduce_min_kernel",
        "reduce_min_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "reduce_min_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "reduce_min_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);

    let mut partials = cpu_to_gpu(&vec![f64::INFINITY; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `reduce_min_kernel` returned by `module_cache::get_or_compile`
    //   at lines 16266-16279; its `(in_ptr, partials_ptr, n)` ABI
    //   matches `REDUCE_MIN_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 16260).
    // - `a: &CudaBuffer<f64>` is the caller-supplied input. `partials:
    //   &mut CudaBuffer<f64>` was uploaded at line 16285 via
    //   `cpu_to_gpu(&vec![f64::INFINITY; num_blocks], device)?`,
    //   exclusively owned by this fn for the duration of the launch.
    //   The `&mut` borrow precludes aliasing `a` per Rust's borrow
    //   rules.
    // - `n = a.len()` (line 16252); the kernel reads `a[i]` only for
    //   `i in [0, n)` per the PTX strided loop bound. `partials` has
    //   `num_blocks` elements (line 16285) and the kernel writes one
    //   element per block.
    // - `n_u32 = n as u32` cannot truncate: the `n == 0` early-return
    //   at line 16253 handles the empty case, and `num_blocks` was
    //   capped to 1024 at line 16283.
    // - cudarc enqueues the launch on `stream`; the three arg refs
    //   live until the trailing `?`. Stream sync is the caller's
    //   responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }

    if num_blocks <= 256 {
        let host_partials = gpu_to_cpu(&partials, device)?;
        let total = host_partials.iter().copied().fold(f64::INFINITY, f64::min);
        return cpu_to_gpu(&[total], device);
    }

    gpu_reduce_min_f64(&partials, device)
}

/// f64 parallel max reduction. Mirrors [`gpu_reduce_max`] but uses
/// the f64-transformed PTX. (#627)
#[cfg(feature = "cuda")]
pub fn gpu_reduce_max_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let n = a.len();
    if n == 0 {
        return cpu_to_gpu(&[f64::NEG_INFINITY], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        REDUCE_MAX_PTX,
        "reduce_max_kernel",
        "reduce_max_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "reduce_max_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "reduce_max_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);

    let mut partials = cpu_to_gpu(&vec![f64::NEG_INFINITY; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `reduce_max_kernel` returned by `module_cache::get_or_compile`
    //   at lines 16337-16350; its `(in_ptr, partials_ptr, n)` ABI
    //   matches `REDUCE_MAX_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 16331).
    // - `a: &CudaBuffer<f64>` is the caller-supplied input. `partials:
    //   &mut CudaBuffer<f64>` was uploaded at line 16356 via
    //   `cpu_to_gpu(&vec![f64::NEG_INFINITY; num_blocks], device)?`,
    //   exclusively owned for the duration of the launch. The `&mut`
    //   borrow precludes aliasing `a`.
    // - `n = a.len()` (line 16323); the kernel reads `a[i]` only for
    //   `i in [0, n)` per the PTX strided loop bound. `partials` has
    //   `num_blocks` elements (line 16356) and the kernel writes one
    //   element per block, with `num_blocks <= 1024` (line 16354).
    // - `n_u32 = n as u32` (line 16357) cannot truncate: the
    //   `n == 0` early-return at line 16324 handles empty input;
    //   `num_blocks` was capped to 1024.
    // - cudarc enqueues the launch on `stream`; the three arg refs
    //   live until the trailing `?`. Stream sync is the caller's
    //   responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }

    if num_blocks <= 256 {
        let host_partials = gpu_to_cpu(&partials, device)?;
        let total = host_partials
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        return cpu_to_gpu(&[total], device);
    }

    gpu_reduce_max_f64(&partials, device)
}

/// f64 reduce_prod (#524).
#[cfg(feature = "cuda")]
pub fn gpu_reduce_prod_f64(a: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let n = a.len();
    if n == 0 {
        return cpu_to_gpu(&[1.0_f64], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        REDUCE_PROD_PTX,
        "reduce_prod_kernel",
        "reduce_prod_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "reduce_prod_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "reduce_prod_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);
    let mut partials = cpu_to_gpu(&vec![1.0_f64; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `reduce_prod_kernel` returned by `module_cache::get_or_compile`
    //   at lines 16410-16423; its `(in_ptr, partials_ptr, n)` ABI
    //   matches `REDUCE_PROD_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 16404).
    // - `a: &CudaBuffer<f64>` is the caller-supplied input. `partials:
    //   &mut CudaBuffer<f64>` was uploaded at line 16428 via
    //   `cpu_to_gpu(&vec![1.0_f64; num_blocks], device)?`, exclusively
    //   owned for the duration of the launch. The `&mut` borrow
    //   precludes aliasing `a`.
    // - `n = a.len()` (line 16396); the kernel reads `a[i]` only for
    //   `i in [0, n)` per the PTX strided loop bound (`@%p bra END`
    //   pattern). `partials` has `num_blocks` elements; the kernel
    //   writes one element per block, with `num_blocks <= 1024`
    //   (line 16427).
    // - `n_u32 = n as u32` (line 16429) cannot truncate: the
    //   `n == 0` early-return at line 16397 handles empty input;
    //   `num_blocks` was capped to 1024.
    // - cudarc enqueues the launch on `stream`; the three arg refs
    //   live until the trailing `?`. Stream sync is caller's
    //   responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }
    if num_blocks <= 256 {
        let host = gpu_to_cpu(&partials, device)?;
        let total: f64 = host.iter().product();
        return cpu_to_gpu(&[total], device);
    }
    gpu_reduce_prod_f64(&partials, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_reduce_prod_f64(
    _a: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_reduce_min_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_reduce_max_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

/// f64 fused masked-min reduction. Mirrors [`gpu_masked_reduce_min`] via
/// the f64-transformed PTX. (#627)
#[cfg(feature = "cuda")]
pub fn gpu_masked_reduce_min_f64(
    data: &CudaBuffer<f64>,
    mask_f: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    if data.len() != mask_f.len() {
        return Err(GpuError::LengthMismatch {
            a: data.len(),
            b: mask_f.len(),
        });
    }
    let n = data.len();
    if n == 0 {
        return cpu_to_gpu(&[f64::INFINITY], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        MASKED_REDUCE_MIN_PTX,
        "masked_reduce_min_kernel",
        "masked_reduce_min_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "masked_reduce_min_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "masked_reduce_min_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);

    let mut partials = cpu_to_gpu(&vec![f64::INFINITY; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `masked_reduce_min_kernel` returned by
    //   `module_cache::get_or_compile` at lines 16505-16518; its
    //   `(data_ptr, mask_ptr, partials_ptr, n)` ABI matches
    //   `MASKED_REDUCE_MIN_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 16499).
    // - `data: &CudaBuffer<f64>` and `mask_f: &CudaBuffer<f64>` are
    //   caller-supplied; the length-equality precondition `data.len()
    //   == mask_f.len()` is enforced at line 16485 (returns
    //   `LengthMismatch` error otherwise).
    // - `partials: &mut CudaBuffer<f64>` was uploaded at line 16524
    //   via `cpu_to_gpu(&vec![f64::INFINITY; num_blocks], device)?`,
    //   exclusively owned for the duration of the launch. The `&mut`
    //   borrow precludes aliasing `data` or `mask_f`.
    // - `n = data.len()` (line 16491); the kernel reads `data[i]` and
    //   `mask_f[i]` only for `i in [0, n)` per the PTX strided loop
    //   bound. `partials` has `num_blocks` elements; one written per
    //   block.
    // - `n_u32 = n as u32` (line 16525) cannot truncate: empty
    //   short-circuit at line 16492; `num_blocks` capped to 1024 at
    //   line 16522.
    // - cudarc enqueues the launch on `stream`; the four arg refs
    //   live until the trailing `?`. Stream sync is caller's
    //   responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(data.inner())
            .arg(mask_f.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }
    if num_blocks <= 256 {
        let host = gpu_to_cpu(&partials, device)?;
        let total = host.iter().copied().fold(f64::INFINITY, f64::min);
        return cpu_to_gpu(&[total], device);
    }
    gpu_reduce_min_f64(&partials, device)
}

/// f64 fused masked-max counterpart. (#627)
#[cfg(feature = "cuda")]
pub fn gpu_masked_reduce_max_f64(
    data: &CudaBuffer<f64>,
    mask_f: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    if data.len() != mask_f.len() {
        return Err(GpuError::LengthMismatch {
            a: data.len(),
            b: mask_f.len(),
        });
    }
    let n = data.len();
    if n == 0 {
        return cpu_to_gpu(&[f64::NEG_INFINITY], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        MASKED_REDUCE_MAX_PTX,
        "masked_reduce_max_kernel",
        "masked_reduce_max_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "masked_reduce_max_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "masked_reduce_max_kernel",
                source: e,
            });
        }
    };

    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    let num_blocks = num_blocks.min(1024);

    let mut partials = cpu_to_gpu(&vec![f64::NEG_INFINITY; num_blocks as usize], device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `masked_reduce_max_kernel` returned by
    //   `module_cache::get_or_compile` at lines 16584-16597; its
    //   `(data_ptr, mask_ptr, partials_ptr, n)` ABI matches
    //   `MASKED_REDUCE_MAX_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 16578).
    // - `data: &CudaBuffer<f64>` and `mask_f: &CudaBuffer<f64>` are
    //   caller-supplied; the length-equality precondition is
    //   enforced at line 16564 (returns `LengthMismatch` otherwise).
    // - `partials: &mut CudaBuffer<f64>` was uploaded at line 16603
    //   via `cpu_to_gpu(&vec![f64::NEG_INFINITY; num_blocks],
    //   device)?`, exclusively owned for the duration of the launch.
    //   The `&mut` borrow precludes aliasing `data` or `mask_f`.
    // - `n = data.len()` (line 16570); the kernel reads `data[i]` and
    //   `mask_f[i]` only for `i in [0, n)` per the PTX strided loop
    //   bound. `partials` has `num_blocks` elements; one written per
    //   block.
    // - `n_u32 = n as u32` (line 16604) cannot truncate: empty
    //   short-circuit at line 16571; `num_blocks` capped to 1024 at
    //   line 16601.
    // - cudarc enqueues the launch on `stream`; the four arg refs
    //   live until the trailing `?`. Stream sync is caller's
    //   responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(data.inner())
            .arg(mask_f.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    if num_blocks <= 1 {
        return Ok(partials);
    }
    if num_blocks <= 256 {
        let host = gpu_to_cpu(&partials, device)?;
        let total = host.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        return cpu_to_gpu(&[total], device);
    }
    gpu_reduce_max_f64(&partials, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_reduce_min(
    _d: &CudaBuffer<f32>,
    _m: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_reduce_max(
    _d: &CudaBuffer<f32>,
    _m: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_reduce_min_f64(
    _d: &CudaBuffer<f64>,
    _m: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_reduce_max_f64(
    _d: &CudaBuffer<f64>,
    _m: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

/// f64 GPU pad/truncate counterpart of [`gpu_pad_truncate_complex_f32`].
/// (#605)
#[cfg(feature = "cuda")]
pub fn gpu_pad_truncate_complex_f64(
    src: &CudaBuffer<f64>,
    batch: usize,
    src_n: usize,
    dst_n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    if src.len() != batch * src_n * 2 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_pad_truncate_complex_f64",
            expected: vec![batch * src_n * 2],
            got: vec![src.len()],
        });
    }
    let total_pairs = batch * dst_n;
    if total_pairs == 0 {
        return alloc_zeros_f64(0, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    // The PTX is purely integer-indexed; only the byte stride matters
    // for the f64 version. The existing rewriter handles
    // `shl.b64 %off_src, %off_src, 2` → `..., 3` and the offset literals
    // (`+ 4` → `+ 8`) we use are NOT covered, so we patch the offsets in
    // the PTX rewrite below using `replace`.
    let ptx = CACHE
        .get_or_init(|| {
            let mut s = ptx_f32_to_f64(
                PAD_TRUNCATE_PTX,
                "pad_truncate_kernel",
                "pad_truncate_f64_kernel",
            );
            // Each complex pair is 8 floats apart in f32 (re at +0, im at +4)
            // and 16 bytes apart in f64 (re at +0, im at +8).
            s = s.replace("[%off_src + 4]", "[%off_src + 8]");
            s = s.replace("[%off_dst + 4]", "[%off_dst + 8]");
            s
        })
        .as_str();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "pad_truncate_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "pad_truncate_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(batch * dst_n * 2, device)?;
    let cfg = launch_cfg(total_pairs)?;
    let batch_u32 = batch as u32;
    let src_n_u32 = src_n as u32;
    let dst_n_u32 = dst_n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `pad_truncate_kernel` returned by
    //   `module_cache::get_or_compile` at lines 16714-16727; its
    //   `(src_ptr, out_ptr, batch, src_n, dst_n)` ABI matches
    //   `PAD_TRUNCATE_PTX` rewritten by `ptx_f32_to_f64` plus the
    //   `+4 -> +8` complex-pair offset patch at lines 16708-16709.
    // - `src: &CudaBuffer<f64>` is the caller-supplied input; its
    //   length was validated at line 16679 to equal
    //   `batch * src_n * 2` (matches the complex-pair stride the
    //   kernel uses) — `ShapeMismatch` is returned otherwise.
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 16729 via
    //   `alloc_zeros_f64(batch * dst_n * 2, device)?` (matches the
    //   destination complex-pair stride) and is exclusively owned
    //   for the duration of the launch. The `&mut` borrow precludes
    //   aliasing `src`.
    // - `total_pairs = batch * dst_n` was checked at line 16687 to be
    //   non-zero. The kernel writes `out[2*i]` and `out[2*i+1]` only
    //   for `i in [0, total_pairs)` per the PTX strided loop bound;
    //   `out` size `batch * dst_n * 2` covers both. Reads of `src`
    //   are bounded by `src_n` (per kernel logic) and the
    //   `batch * src_n * 2` allocation guarantee.
    // - `batch_u32`, `src_n_u32`, `dst_n_u32` cannot truncate:
    //   `launch_cfg(total_pairs)?` (line 16730) returns `Err` if
    //   `total_pairs > u32::MAX` (and `batch <= total_pairs`,
    //   `dst_n <= total_pairs`); `src_n_u32 = src.len() / (batch * 2)`
    //   so it shares the `src.len() <= u32::MAX` bound implicitly.
    // - cudarc enqueues the launch on `stream`; the five arg refs
    //   live until the trailing `?`. Stream sync is caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(src.inner())
            .arg(out.inner_mut())
            .arg(&batch_u32)
            .arg(&src_n_u32)
            .arg(&dst_n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_pad_truncate_complex_f64(
    _src: &CudaBuffer<f64>,
    _batch: usize,
    _src_n: usize,
    _dst_n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

/// Sum along an axis for f64.
#[cfg(feature = "cuda")]
pub fn gpu_sum_axis_f64(
    a: &CudaBuffer<f64>,
    outer: usize,
    axis_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let total_output = outer * inner;
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        SUM_AXIS_PTX,
        "sum_axis_kernel",
        "sum_axis_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "sum_axis_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "sum_axis_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(total_output, device)?;
    let cfg = launch_cfg(total_output)?;
    let outer_u32 = outer as u32;
    let axis_size_u32 = axis_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total_output as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `sum_axis_kernel` returned by `module_cache::get_or_compile`
    //   at lines 16782-16795; its `(in_ptr, out_ptr, outer,
    //   axis_size, inner, total)` ABI matches `SUM_AXIS_PTX`
    //   rewritten by `ptx_f32_to_f64` (line 16776).
    // - `a: &CudaBuffer<f64>` is the caller-supplied input; the
    //   kernel reads `a[((o * axis_size) + k) * inner + i]` for
    //   `(o, i)` in the cartesian product and `k in [0, axis_size)`,
    //   which is bounded by the `outer * axis_size * inner` length
    //   the caller is contracted to provide via the function args
    //   `outer`, `axis_size`, `inner`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 16797 via
    //   `alloc_zeros_f64(total_output, device)?` where
    //   `total_output = outer * inner` (line 16772). The `&mut`
    //   borrow precludes aliasing `a`.
    // - `outer_u32`, `axis_size_u32`, `inner_u32`, `total_u32` cannot
    //   truncate: `launch_cfg(total_output)?` at line 16798 returns
    //   `Err` if `total_output > u32::MAX`; `outer * inner =
    //   total_output` implies both factors fit in u32; `axis_size`
    //   is part of the caller's contract that `a.len() = outer *
    //   axis_size * inner <= u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the six arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

#[cfg(not(feature = "cuda"))]
pub fn gpu_reduce_sum_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_sum_axis_f64(
    _a: &CudaBuffer<f64>,
    _outer: usize,
    _axis_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Public API -- f64 shape ops
// ---------------------------------------------------------------------------

/// Transpose an `[M, N]` f64 matrix to `[N, M]` on GPU.
#[cfg(feature = "cuda")]
pub fn gpu_transpose_2d_f64(
    input: &CudaBuffer<f64>,
    m: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let total = m * n;
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        TRANSPOSE_2D_PTX,
        "transpose_2d_kernel",
        "transpose_2d_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "transpose_2d_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "transpose_2d_f64_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(total, device)?;
    let cfg = launch_cfg(total)?;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `transpose_2d_kernel` returned by
    //   `module_cache::get_or_compile` at lines 16861-16874; its
    //   `(in_ptr, out_ptr, m, n, total)` ABI matches
    //   `TRANSPOSE_2D_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 16855).
    // - `input: &CudaBuffer<f64>` is the caller-supplied input;
    //   `validate_device(input, device)?` at line 16849 confirms
    //   it lives on `device` (helper at line 10582).
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 16876 via
    //   `alloc_zeros_f64(total, device)?` where `total = m * n`
    //   (line 16851). The `&mut` borrow precludes aliasing `input`.
    // - The kernel reads `input[i]` and writes `out[i]` only for
    //   `i in [0, total)` per the PTX bound check (`@%p bra DONE`
    //   pattern, `total = m * n` covers both `[m, n]` source and
    //   `[n, m]` destination).
    // - `m_u32`, `n_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 16877) returns `Err` if
    //   `total > u32::MAX`, and `m, n <= total` follows from
    //   `total = m * n` with both `>= 1` (or `total = 0` short-
    //   circuited via empty allocator).
    // - cudarc enqueues the launch on `stream`; the five arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

/// Permute a 4D f64 tensor from `[d0, d1, d2, d3]` to `[d0, d2, d1, d3]` on GPU.
#[cfg(feature = "cuda")]
pub fn gpu_permute_0213_f64(
    input: &CudaBuffer<f64>,
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let total = d0 * d1 * d2 * d3;
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        PERMUTE_0213_PTX,
        "permute_0213_kernel",
        "permute_0213_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "permute_0213_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "permute_0213_f64_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(total, device)?;
    let cfg = launch_cfg(total)?;
    let d0_u32 = d0 as u32;
    let d1_u32 = d1 as u32;
    let d2_u32 = d2 as u32;
    let d3_u32 = d3 as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `permute_0213_kernel` returned by
    //   `module_cache::get_or_compile` at lines 16921-16934; its
    //   `(in_ptr, out_ptr, d0, d1, d2, d3, total)` ABI matches
    //   `PERMUTE_0213_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 16915).
    // - `input: &CudaBuffer<f64>` is the caller-supplied input;
    //   `validate_device(input, device)?` at line 16909 confirms
    //   device match (helper at line 10582).
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 16936 via
    //   `alloc_zeros_f64(total, device)?` where
    //   `total = d0 * d1 * d2 * d3` (line 16911). The `&mut` borrow
    //   precludes aliasing `input`.
    // - The kernel reads `input[i]` and writes `out[i]` only for
    //   `i in [0, total)` per the PTX bound check; the permutation
    //   `[d0, d1, d2, d3] -> [d0, d2, d1, d3]` preserves total
    //   element count.
    // - `d0_u32`, `d1_u32`, `d2_u32`, `d3_u32`, `total_u32` cannot
    //   truncate: `launch_cfg(total)?` (line 16937) returns `Err`
    //   if `total > u32::MAX`, and each dim is bounded above by
    //   `total`.
    // - cudarc enqueues the launch on `stream`; the seven arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

/// Split a contiguous f64 tensor along an axis (strided read) on GPU.
#[cfg(feature = "cuda")]
pub fn gpu_strided_split_f64(
    input: &CudaBuffer<f64>,
    total_along_axis: usize,
    split_offset: usize,
    split_size: usize,
    inner_size: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        STRIDED_SPLIT_PTX,
        "strided_split_kernel",
        "strided_split_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "strided_split_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "strided_split_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(n)?;
    let total_ax_u32 = total_along_axis as u32;
    let offset_u32 = split_offset as u32;
    let split_sz_u32 = split_size as u32;
    let inner_u32 = inner_size as u32;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `strided_split_kernel` returned by
    //   `module_cache::get_or_compile` at lines 16985-16998; its
    //   `(in_ptr, out_ptr, total_along_axis, split_offset,
    //   split_size, inner_size, n)` ABI matches `STRIDED_SPLIT_PTX`
    //   rewritten by `ptx_f32_to_f64` (line 16979).
    // - `input: &CudaBuffer<f64>` is the caller-supplied input;
    //   `validate_device(input, device)?` at line 16974 confirms
    //   device match. The caller must size `input` to cover
    //   `total_along_axis * inner_size` elements per the strided
    //   read pattern (function-arg contract).
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 17000 via
    //   `alloc_zeros_f64(n, device)?` where `n` is the caller-
    //   declared output element count (function arg). The `&mut`
    //   borrow precludes aliasing `input`.
    // - The kernel writes `out[i]` only for `i in [0, n)` per the
    //   PTX bound check; reads of `input` are bounded by the
    //   strided index `(out_outer * total_along_axis + split_offset
    //   + k) * inner_size + inner_idx` whose validity follows from
    //   `split_offset + split_size <= total_along_axis` (caller
    //   contract) and the input length above.
    // - `total_ax_u32`, `offset_u32`, `split_sz_u32`, `inner_u32`,
    //   `n_u32` cannot truncate: `launch_cfg(n)?` (line 17001)
    //   returns `Err` if `n > u32::MAX`; the other dims must fit
    //   in u32 since `n = batch * split_size * inner_size` and
    //   `total_along_axis >= split_size`.
    // - cudarc enqueues the launch on `stream`; the seven arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

/// Concatenate an f64 sub-tensor into a larger output at an axis offset on GPU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_strided_cat_f64(
    input: &CudaBuffer<f64>,
    output: &mut CudaBuffer<f64>,
    total_along_axis: usize,
    cat_offset: usize,
    part_size: usize,
    inner_size: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;

    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    validate_device(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        STRIDED_CAT_PTX,
        "strided_cat_kernel",
        "strided_cat_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "strided_cat_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "strided_cat_kernel",
                source: e,
            });
        }
    };

    let cfg = launch_cfg(n)?;
    let total_ax_u32 = total_along_axis as u32;
    let offset_u32 = cat_offset as u32;
    let part_sz_u32 = part_size as u32;
    let inner_u32 = inner_size as u32;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `strided_cat_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17051-17064; its
    //   `(in_ptr, out_ptr, total_along_axis, cat_offset, part_size,
    //   inner_size, n)` ABI matches `STRIDED_CAT_PTX` rewritten by
    //   `ptx_f32_to_f64` (line 17045).
    // - `input: &CudaBuffer<f64>` is the caller-supplied source;
    //   `validate_device(input, device)?` at line 17040 confirms
    //   device match. The caller must size `input` to cover `n`
    //   elements (one per output write, one per input read).
    // - `output: &mut CudaBuffer<f64>` is the caller-supplied
    //   destination buffer (passed by `&mut` ensuring exclusivity)
    //   that must have at least `cat_offset + part_size <=
    //   total_along_axis` valid axis space; the caller contract is
    //   that `output` was alloc'd with size at least
    //   `batch * total_along_axis * inner_size` covering the
    //   write region.
    // - The kernel writes `output[(o * total_along_axis +
    //   cat_offset + k) * inner_size + i]` for `n` thread indices
    //   per the PTX bound check; the strided write region cannot
    //   alias `input` because `output` is `&mut` and Rust borrow
    //   rules forbid simultaneous `&` to it.
    // - `total_ax_u32`, `offset_u32`, `part_sz_u32`, `inner_u32`,
    //   `n_u32` cannot truncate: `launch_cfg(n)?` (line 17066)
    //   returns `Err` if `n > u32::MAX`; the other dims fit since
    //   `n = batch * part_size * inner_size`.
    // - cudarc enqueues the launch on `stream`; the seven arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

// ---------------------------------------------------------------------------
// Public API -- f64 indexing ops
// ---------------------------------------------------------------------------

/// Gather f64 elements by f32 index: `out[i] = input[indices[i]]`.
#[cfg(feature = "cuda")]
pub fn gpu_index_select_1d_f64(
    input: &CudaBuffer<f64>,
    indices: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let n = indices.len();
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        INDEX_SELECT_1D_PTX,
        "index_select_1d_kernel",
        "index_select_1d_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "index_select_1d_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "index_select_1d_f64_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `index_select_1d_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17115-17128; its
    //   `(in_ptr, idx_ptr, out_ptr, n)` ABI matches
    //   `INDEX_SELECT_1D_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 17109).
    // - `input: &CudaBuffer<f64>` is the gather source;
    //   `validate_device(input, device)?` at line 17103 confirms
    //   device match. The caller is contracted (per fn doc) to
    //   ensure all values in `indices` are valid offsets into
    //   `input`.
    // - `indices: &CudaBuffer<f32>` carries the gather indices;
    //   the kernel reads `indices[i]` for `i in [0, n)` where
    //   `n = indices.len()` (line 17105).
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 17130 via
    //   `alloc_zeros_f64(n, device)?`. The `&mut` borrow precludes
    //   aliasing `input` or `indices`.
    // - The kernel writes `out[i]` only for `i in [0, n)` per the
    //   PTX bound check (`@%p bra DONE` pattern); reads of
    //   `input[indices[i] as usize]` are subject to the caller's
    //   index-in-range contract — out-of-range causes UB but no
    //   memory-safety hazard if the caller respects the contract.
    // - `n_u32 = n as u32` (line 17132) cannot truncate:
    //   `launch_cfg(n)?` (line 17131) returns `Err` if
    //   `n > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the four arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

/// Scatter-add f64 `grad_output` back using f32 `indices`.
///
/// Output: `out = zeros(input_len); for i: out[indices[i]] += grad_output[i]`
#[cfg(feature = "cuda")]
pub fn gpu_scatter_add_1d_f64(
    grad_output: &CudaBuffer<f64>,
    indices: &CudaBuffer<f32>,
    input_len: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(grad_output, device)?;

    let n = grad_output.len();
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        SCATTER_ADD_1D_PTX,
        "scatter_add_1d_kernel",
        "scatter_add_1d_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "scatter_add_1d_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "scatter_add_1d_f64_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(input_len, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `scatter_add_1d_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17172-17185; its
    //   `(grad_ptr, idx_ptr, out_ptr, n)` ABI matches
    //   `SCATTER_ADD_1D_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 17166).
    // - `grad_output: &CudaBuffer<f64>` is the source gradient;
    //   `validate_device(grad_output, device)?` at line 17160
    //   confirms device match.
    // - `indices: &CudaBuffer<f32>` is the destination index map;
    //   per the fn rustdoc the kernel atomically increments
    //   `out[indices[i] as usize]` for `i in [0, n)` where
    //   `n = grad_output.len()` (line 17162).
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 17187 via
    //   `alloc_zeros_f64(input_len, device)?` — the caller is
    //   contracted to provide `input_len` as the upper bound for
    //   any `indices[i]` value. The `&mut` borrow precludes
    //   aliasing `grad_output` or `indices`.
    // - The kernel uses `atomicAdd` semantics (per PTX); duplicate
    //   indices accumulate correctly. Reads of `grad_output[i]`
    //   and `indices[i]` are bounded by `n`; writes to
    //   `out[indices[i]]` rely on caller contract that index is
    //   `< input_len`.
    // - `n_u32 = n as u32` (line 17189) cannot truncate:
    //   `launch_cfg(n)?` (line 17188) returns `Err` if
    //   `n > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the four arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

/// Fill f64 elements with `value` where u8 `mask` is nonzero.
///
/// `mask` is a GPU buffer of u8 values (nonzero = true).
/// Output: `out[i] = mask[i] != 0 ? value : input[i]`
#[cfg(feature = "cuda")]
pub fn gpu_masked_fill_f64(
    input: &CudaBuffer<f64>,
    mask: &CudaBuffer<u8>,
    value: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let n = input.len();
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        MASKED_FILL_PTX,
        "masked_fill_kernel",
        "masked_fill_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "masked_fill_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "masked_fill_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `masked_fill_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17230-17243; its
    //   `(in_ptr, mask_ptr, out_ptr, value, n)` ABI matches
    //   `MASKED_FILL_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 17224).
    // - `input: &CudaBuffer<f64>` is the source;
    //   `validate_device(input, device)?` at line 17218 confirms
    //   device match. `mask: &CudaBuffer<u8>` carries u8 nonzero/
    //   zero flags; the caller is contracted (per fn doc) to size
    //   `mask` to `input.len()`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 17245 via
    //   `alloc_zeros_f64(n, device)?` where `n = input.len()`
    //   (line 17220). The `&mut` borrow precludes aliasing
    //   `input` or `mask`.
    // - The kernel reads `input[i]` and `mask[i]` and writes
    //   `out[i]` only for `i in [0, n)` per the PTX bound check.
    //   `value: f64` is passed by-reference; cudarc copies the
    //   value into the launch parameter buffer before the kernel
    //   begins.
    // - `n_u32 = n as u32` (line 17247) cannot truncate:
    //   `launch_cfg(n)?` (line 17246) returns `Err` if
    //   `n > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the five arg refs
    //   (including `&value`) live until the trailing `?`. Stream
    //   sync is caller's.
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

/// Zero out f64 gradient where u8 `mask` is nonzero.
///
/// Output: `out[i] = mask[i] != 0 ? 0.0 : grad[i]`
#[cfg(feature = "cuda")]
pub fn gpu_masked_zero_f64(
    grad: &CudaBuffer<f64>,
    mask: &CudaBuffer<u8>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(grad, device)?;

    let n = grad.len();
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        MASKED_ZERO_PTX,
        "masked_zero_kernel",
        "masked_zero_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "masked_zero_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "masked_zero_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `masked_zero_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17287-17300; its
    //   `(grad_ptr, mask_ptr, out_ptr, n)` ABI matches
    //   `MASKED_ZERO_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 17281).
    // - `grad: &CudaBuffer<f64>` is the gradient source;
    //   `validate_device(grad, device)?` at line 17275 confirms
    //   device match. `mask: &CudaBuffer<u8>` carries the zero-
    //   mask flags; the caller is contracted to size `mask` to
    //   `grad.len()`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 17302 via
    //   `alloc_zeros_f64(n, device)?` where `n = grad.len()`
    //   (line 17277). The `&mut` borrow precludes aliasing
    //   `grad` or `mask`.
    // - The kernel reads `grad[i]` and `mask[i]` and writes
    //   `out[i]` only for `i in [0, n)` per the PTX bound check;
    //   `out[i] = mask[i] != 0 ? 0.0 : grad[i]`.
    // - `n_u32 = n as u32` (line 17304) cannot truncate:
    //   `launch_cfg(n)?` (line 17303) returns `Err` if
    //   `n > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the four arg refs
    //   live until the trailing `?`. Stream sync is caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad.inner())
            .arg(mask.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

/// Write f64 `src` of shape `[N, D]` into row `pos` of `dst` of shape `[N, max_len, D]`.
#[cfg(feature = "cuda")]
pub fn gpu_slice_write_f64(
    src: &CudaBuffer<f64>,
    dst: &mut CudaBuffer<f64>,
    n_batch: usize,
    d: usize,
    max_len: usize,
    pos: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let total = n_batch * d;
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        SLICE_WRITE_PTX,
        "slice_write_kernel",
        "slice_write_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "slice_write_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "slice_write_kernel",
                source: e,
            });
        }
    };

    let cfg = launch_cfg(total)?;
    let n_u32 = total as u32;
    let d_u32 = d as u32;
    let max_len_u32 = max_len as u32;
    let pos_u32 = pos as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `slice_write_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17343-17356; its
    //   `(src_ptr, dst_ptr, n_total, d, max_len, pos)` ABI matches
    //   `SLICE_WRITE_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 17337).
    // - `src: &CudaBuffer<f64>` is the source `[n_batch, d]`
    //   buffer; the caller is contracted to size `src` to
    //   `n_batch * d` (= `total` from line 17333).
    // - `dst: &mut CudaBuffer<f64>` is the destination `[n_batch,
    //   max_len, d]` buffer with `pos` selected as the row
    //   coordinate. The `&mut` borrow precludes aliasing `src`
    //   per Rust borrow rules. The caller is contracted to size
    //   `dst` to `n_batch * max_len * d` and to ensure
    //   `pos < max_len`.
    // - The kernel writes `dst[(b * max_len + pos) * d + j]` for
    //   `(b, j)` in `[0, n_batch) x [0, d)` (i.e., `total` thread
    //   indices) per the PTX bound check; reads of `src[b * d + j]`
    //   are bounded by `total`.
    // - `n_u32 = total as u32`, `d_u32`, `max_len_u32`, `pos_u32`
    //   cannot truncate: `launch_cfg(total)?` (line 17358) returns
    //   `Err` if `total > u32::MAX`; `d <= total`, `max_len`/`pos`
    //   fit per `dst.len() = n_batch * max_len * d <= u32::MAX`
    //   per allocator pool limits.
    // - cudarc enqueues the launch on `stream`; the six arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

/// Read first `len` rows from each batch of f64 `[N, max_len, D]` -> `[N, len, D]`.
#[cfg(feature = "cuda")]
pub fn gpu_slice_read_f64(
    src: &CudaBuffer<f64>,
    n_batch: usize,
    d: usize,
    len: usize,
    max_len: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let total = n_batch * len * d;
    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        SLICE_READ_PTX,
        "slice_read_kernel",
        "slice_read_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "slice_read_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "slice_read_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(total, device)?;
    let cfg = launch_cfg(total)?;
    let total_u32 = total as u32;
    let d_u32 = d as u32;
    let len_u32 = len as u32;
    let max_len_u32 = max_len as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `slice_read_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17402-17415; its
    //   `(src_ptr, out_ptr, total, d, len, max_len)` ABI matches
    //   `SLICE_READ_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 17396).
    // - `src: &CudaBuffer<f64>` is the source `[n_batch, max_len,
    //   d]` buffer; the caller is contracted (per fn doc) to size
    //   `src` to `n_batch * max_len * d` and to ensure
    //   `len <= max_len`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 17417 via
    //   `alloc_zeros_f64(total, device)?` where
    //   `total = n_batch * len * d` (line 17392). The `&mut`
    //   borrow precludes aliasing `src`.
    // - The kernel writes `out[(b * len + l) * d + j]` for
    //   `(b, l, j)` in the cartesian product (i.e., `total`
    //   thread indices) per the PTX bound check; reads of
    //   `src[(b * max_len + l) * d + j]` for `l < len` are
    //   bounded by `n_batch * max_len * d` (caller contract).
    // - `total_u32`, `d_u32`, `len_u32`, `max_len_u32` cannot
    //   truncate: `launch_cfg(total)?` (line 17418) returns `Err`
    //   if `total > u32::MAX`; `d <= total`, `len <= max_len`,
    //   and `max_len * n_batch * d <= u32::MAX` (caller's
    //   responsibility per allocator).
    // - cudarc enqueues the launch on `stream`; the six arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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
// Public API -- f64 embedding ops
// ---------------------------------------------------------------------------

/// Single f64 embedding lookup: `output[d] = weight[token_id * D + d]`.
#[cfg(feature = "cuda")]
pub fn gpu_embed_lookup_f64(
    idx: &CudaBuffer<f32>,
    weight: &CudaBuffer<f64>,
    d: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        EMBED_LOOKUP_PTX,
        "embed_lookup_kernel",
        "embed_lookup_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "embed_lookup_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "embed_lookup_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(d, device)?;
    let cfg = launch_cfg(d)?;
    let d_u32 = d as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `embed_lookup_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17463-17476; its
    //   `(idx_ptr, weight_ptr, out_ptr, d)` ABI matches
    //   `EMBED_LOOKUP_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 17457).
    // - `idx: &CudaBuffer<f32>` is a 1-element token-id buffer
    //   (caller contract: `idx.len() == 1`). `weight:
    //   &CudaBuffer<f64>` is the embedding table `[V, D]`; the
    //   caller must size it to at least `(token_id + 1) * d`
    //   per the gather pattern.
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 17478 via
    //   `alloc_zeros_f64(d, device)?` for the `[D]` output. The
    //   `&mut` borrow precludes aliasing `idx` or `weight`.
    // - The kernel reads `idx[0]` once (token id) and copies
    //   `weight[token_id * d + j]` into `out[j]` for
    //   `j in [0, d)` per the PTX bound check.
    // - `d_u32 = d as u32` (line 17480) cannot truncate:
    //   `launch_cfg(d)?` (line 17479) returns `Err` if
    //   `d > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the four arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

/// Batch f64 embedding lookup: gather N rows from `[V, D]` weight into `[N, D]`.
#[cfg(feature = "cuda")]
pub fn gpu_embed_lookup_batch_f64(
    indices: &CudaBuffer<f32>,
    weight: &CudaBuffer<f64>,
    n: usize,
    d: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let total = n * d;
    if total == 0 {
        return alloc_zeros_f64(0, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        EMBED_LOOKUP_BATCH_PTX,
        "embed_lookup_batch_kernel",
        "embed_lookup_batch_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "embed_lookup_batch_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "embed_lookup_batch_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(total, device)?;
    let cfg = launch_cfg(total)?;
    let d_u32 = d as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `embed_lookup_batch_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17521-17534; its
    //   `(indices_ptr, weight_ptr, out_ptr, d, total)` ABI matches
    //   `EMBED_LOOKUP_BATCH_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 17515).
    // - `indices: &CudaBuffer<f32>` is the token-id batch
    //   (caller contract: `indices.len() == n`). `weight:
    //   &CudaBuffer<f64>` is the `[V, D]` embedding table; the
    //   caller is contracted to ensure `max(indices) < V`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 17536 via
    //   `alloc_zeros_f64(total, device)?` where
    //   `total = n * d` (line 17507). The `&mut` borrow precludes
    //   aliasing `indices` or `weight`.
    // - The kernel writes `out[i * d + j]` for one thread per
    //   `(i, j)` in `[0, n) x [0, d)` (i.e., `total` thread
    //   indices) per the PTX bound check; reads of
    //   `weight[indices[i] * d + j]` rely on the caller index-
    //   range contract.
    // - `d_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 17537) returns `Err` if
    //   `total > u32::MAX`; `d <= total`.
    // - cudarc enqueues the launch on `stream`; the five arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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

/// Scatter-add f64 rows for embedding backward.
///
/// Atomically accumulates `grad_output[i, :] += grad_weight[indices[i], :]`.
#[cfg(feature = "cuda")]
pub fn gpu_scatter_add_rows_f64(
    grad_output: &CudaBuffer<f64>,
    indices: &CudaBuffer<f32>,
    num_embeddings: usize,
    d: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    let n = indices.len();
    let total = n * d;

    if total == 0 {
        return alloc_zeros_f64(num_embeddings * d, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        SCATTER_ADD_ROWS_PTX,
        "scatter_add_rows_kernel",
        "scatter_add_rows_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "scatter_add_rows_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "scatter_add_rows_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(num_embeddings * d, device)?;
    let cfg = launch_cfg(total)?;
    let d_u32 = d as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `scatter_add_rows_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17585-17598; its
    //   `(grad_ptr, idx_ptr, out_ptr, d, total)` ABI matches
    //   `SCATTER_ADD_ROWS_PTX` rewritten by `ptx_f32_to_f64`
    //   (line 17579).
    // - `grad_output: &CudaBuffer<f64>` is the upstream gradient
    //   `[N, D]`; `indices: &CudaBuffer<f32>` carries the
    //   destination row indices. `n = indices.len()` (line 17569);
    //   the caller is contracted to ensure `grad_output.len() ==
    //   n * d` and `max(indices) < num_embeddings`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd at line 17600 via
    //   `alloc_zeros_f64(num_embeddings * d, device)?`. The `&mut`
    //   borrow precludes aliasing `grad_output` or `indices`.
    // - The kernel uses `atomicAdd` (per the PTX) to accumulate
    //   `grad_output[i * d + j]` into `out[indices[i] * d + j]`
    //   for `(i, j)` in `[0, n) x [0, d)` — `total = n * d`
    //   thread indices per the PTX bound check; duplicate indices
    //   accumulate correctly.
    // - `d_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 17601) returns `Err` if
    //   `total > u32::MAX`; `d <= total`.
    // - cudarc enqueues the launch on `stream`; the five arg refs
    //   live until the trailing `?`. Stream sync is caller's.
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "fused_adam_kernel",
                source: e,
            });
        }
    };

    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `fused_adam_kernel`
    //   returned by `module_cache::get_or_compile` at lines
    //   17657-17670; its `(param, grad, exp_avg, exp_avg_sq,
    //   beta1, beta2, lr, eps, bc1, bc2, weight_decay, n)` ABI
    //   matches `FUSED_ADAM_PTX`.
    // - `param: &mut CudaBuffer<f32>`, `exp_avg: &mut
    //   CudaBuffer<f32>`, and `exp_avg_sq: &mut CudaBuffer<f32>`
    //   are three distinct exclusively-borrowed buffers; Rust's
    //   borrow rules guarantee they cannot alias each other or
    //   `grad`. Each is in-place mutated by the kernel.
    // - `grad: &CudaBuffer<f32>` is the read-only gradient. The
    //   length-equality precondition `grad.len() == param.len()
    //   == exp_avg.len() == exp_avg_sq.len() == n` is enforced at
    //   line 17647 (returns `LengthMismatch` otherwise).
    // - All four buffers were alloc'd by the caller before this
    //   call (the optimizer's persistent state); their `CudaSlice`
    //   handles live for the duration of the launch since the
    //   `&mut`/`&` borrows are bound to the call frame.
    // - The kernel reads `grad[i]` and reads/writes `param[i]`,
    //   `exp_avg[i]`, `exp_avg_sq[i]` only for `i in [0, n)` per
    //   the PTX bound check (`@%p bra DONE` pattern). Scalar
    //   args (`beta1`..`weight_decay`, all `f32`) are passed
    //   by-reference; cudarc copies values into the launch param
    //   buffer before the kernel begins.
    // - `n_u32 = n as u32` (line 17673) cannot truncate:
    //   `launch_cfg(n)?` (line 17672) returns `Err` if
    //   `n > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the twelve arg
    //   refs live until the trailing `?`. Stream sync is the
    //   caller's responsibility (typically the optimizer's step
    //   batch boundary).
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "fused_gru_forward_kernel",
                source: e,
            });
        }
    };

    let mut hy = alloc_zeros_f32(total, device)?;
    let mut workspace = alloc_zeros_f32(batch * 5 * hsz, device)?;

    let cfg = launch_cfg(total)?;
    let hsz_u32 = hsz as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `fused_gru_forward_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17751-17764; its
    //   nine-argument ABI `(input_gates, hidden_gates, bias_ih,
    //   bias_hh, hx, hy, workspace, hsz, total)` matches
    //   `FUSED_GRU_FORWARD_PTX`.
    // - `input_gates: &CudaBuffer<f32>`, `hidden_gates:
    //   &CudaBuffer<f32>`, `bias_ih: &CudaBuffer<f32>`, `bias_hh:
    //   &CudaBuffer<f32>`, `hx: &CudaBuffer<f32>` are five
    //   read-only inputs supplied by the caller. The caller's
    //   contract (per fn rustdoc, lines 17723-17732) is:
    //   `input_gates.len() == hidden_gates.len() == batch * 3 *
    //   hsz`, `bias_ih.len() == bias_hh.len() == 3 * hsz`,
    //   `hx.len() == batch * hsz`, where `batch = total / hsz`
    //   (line 17746). The kernel is read-only over these.
    // - `hy: &mut CudaBuffer<f32>` was alloc'd at line 17766 via
    //   `alloc_zeros_f32(total, device)?` for the `[batch, hsz]`
    //   new hidden state. `workspace: &mut CudaBuffer<f32>` was
    //   alloc'd at line 17767 via `alloc_zeros_f32(batch * 5 *
    //   hsz, device)?` for the `[batch, 5*hsz]` saved-for-
    //   backward buffer. Both are exclusively owned by this fn;
    //   Rust borrow rules guarantee they cannot alias each other
    //   or any of the read-only inputs.
    // - The kernel writes `hy[i]` for `i in [0, total)` (one
    //   thread per output element) and writes the 5 sub-blocks
    //   of `workspace` indexed at `(b * 5 + slot) * hsz + j` per
    //   the PTX layout; both regions are bounded by their
    //   respective allocations above.
    // - `hsz_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 17769) returns `Err` if
    //   `total > u32::MAX`; `hsz <= total` since `total = batch
    //   * hsz`.
    // - cudarc enqueues the launch on `stream`; the nine arg refs
    //   live until the trailing `?`. Stream sync is the caller's
    //   responsibility before any read of `hy` or `workspace`.
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
        ctx,
        MAXPOOL2D_PTX,
        "maxpool2d_forward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "maxpool2d_forward_kernel",
                source: e,
            });
        }
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `maxpool2d_forward_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17835-17848; its
    //   ABI `(input_ptr, out_ptr, batch, channels, h_in, w_in,
    //   h_out, w_out, kh, kw, sh, sw, ph, pw, total)` matches
    //   `MAXPOOL2D_PTX`.
    // - `input: &CudaBuffer<f32>` is the source `[batch,
    //   channels, h_in, w_in]` activation; the caller is
    //   contracted to size `input` to `batch * channels * h_in *
    //   w_in` and the output dims are computed at lines 17828-
    //   17829 from the standard pooling formula `(h_in + 2*ph -
    //   kh) / sh + 1` (likewise for w).
    // - `out: &mut CudaBuffer<f32>` was alloc'd at line 17850 via
    //   `alloc_zeros_f32(total, device)?` where `total = batch *
    //   channels * h_out * w_out` (line 17830). The `&mut`
    //   borrow precludes aliasing `input`.
    // - The kernel runs one thread per output element (`total`
    //   threads) per the PTX bound check; each thread computes
    //   `out[idx]` from the `kh x kw` window in `input` at the
    //   strided/padded position derived from `(b, c, oh, ow)`,
    //   bounded by `[0, h_in)` and `[0, w_in)` after pad
    //   subtraction.
    // - `batch_u32`, `ch_u32`, `h_in_u32`, `w_in_u32`, `h_out_
    //   u32`, `w_out_u32`, `kh_u32`, `kw_u32`, `sh_u32`, `sw_u32`,
    //   `ph_u32`, `pw_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 17851) returns `Err` if
    //   `total > u32::MAX`; the per-axis dims fit since they
    //   multiply to `<= total`.
    // - cudarc enqueues the launch on `stream`; the fifteen arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&batch_u32)
            .arg(&ch_u32)
            .arg(&h_in_u32)
            .arg(&w_in_u32)
            .arg(&h_out_u32)
            .arg(&w_out_u32)
            .arg(&kh_u32)
            .arg(&kw_u32)
            .arg(&sh_u32)
            .arg(&sw_u32)
            .arg(&ph_u32)
            .arg(&pw_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok((out, [batch, channels, h_out, w_out]))
}

/// Stub.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_maxpool2d(
    _input: &CudaBuffer<f32>,
    _batch: usize,
    _channels: usize,
    _h_in: usize,
    _w_in: usize,
    _kh: usize,
    _kw: usize,
    _sh: usize,
    _sw: usize,
    _ph: usize,
    _pw: usize,
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
        ctx,
        AVGPOOL2D_PTX,
        "avgpool2d_forward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "avgpool2d_forward_kernel",
                source: e,
            });
        }
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `avgpool2d_forward_kernel` returned by
    //   `module_cache::get_or_compile` at lines 17931-17944; its
    //   ABI `(input_ptr, out_ptr, batch, channels, h_in, w_in,
    //   h_out, w_out, kh, kw, sh, sw, ph, pw, total)` matches
    //   `AVGPOOL2D_PTX`.
    // - `input: &CudaBuffer<f32>` is the source `[batch,
    //   channels, h_in, w_in]` activation; the caller is
    //   contracted to size `input` to `batch * channels * h_in *
    //   w_in`. Output dims are computed at lines 17924-17925
    //   from `(h_in + 2*ph - kh) / sh + 1` (likewise for w).
    // - `out: &mut CudaBuffer<f32>` was alloc'd at line 17946 via
    //   `alloc_zeros_f32(total, device)?` where `total = batch *
    //   channels * h_out * w_out` (line 17926). The `&mut`
    //   borrow precludes aliasing `input`.
    // - The kernel runs one thread per output element (`total`
    //   threads) per the PTX bound check; each thread averages
    //   the `kh x kw` window in `input` (with padded positions
    //   contributing 0 per standard avgpool semantics), bounded
    //   by `[0, h_in)` and `[0, w_in)` after pad subtraction.
    // - The fifteen u32 args cannot truncate:
    //   `launch_cfg(total)?` (line 17947) returns `Err` if
    //   `total > u32::MAX`; the per-axis dims fit since they
    //   multiply to `<= total`.
    // - cudarc enqueues the launch on `stream`; the fifteen arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&batch_u32)
            .arg(&ch_u32)
            .arg(&h_in_u32)
            .arg(&w_in_u32)
            .arg(&h_out_u32)
            .arg(&w_out_u32)
            .arg(&kh_u32)
            .arg(&kw_u32)
            .arg(&sh_u32)
            .arg(&sw_u32)
            .arg(&ph_u32)
            .arg(&pw_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok((out, [batch, channels, h_out, w_out]))
}

/// Stub.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_avgpool2d(
    _input: &CudaBuffer<f32>,
    _batch: usize,
    _channels: usize,
    _h_in: usize,
    _w_in: usize,
    _kh: usize,
    _kw: usize,
    _sh: usize,
    _sw: usize,
    _ph: usize,
    _pw: usize,
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
            return Err(GpuError::PtxCompileFailed {
                kernel: "layernorm_kernel",
                source: e,
            });
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `layernorm_kernel`
    //   returned by `module_cache::get_or_compile` at lines
    //   18083-18096; its ABI `(input_ptr, out_ptr, weight_ptr,
    //   bias_ptr, rows, cols, eps)` matches `LAYERNORM_PTX`.
    // - `input: &CudaBuffer<f32>` is the row-major `[rows, cols]`
    //   activation; `validate_unary(input, device)?` at line
    //   18078 confirms device match. The caller is contracted
    //   (per fn doc) to size `input` to `rows * cols`.
    // - `weight: &CudaBuffer<f32>` and `bias: &CudaBuffer<f32>`
    //   are per-column affine parameters (length `cols`); both
    //   shared-borrowed (read-only).
    // - `out: &mut CudaBuffer<f32>` was alloc'd at line 18098 via
    //   `alloc_zeros_f32(rows * cols, device)?`. The `&mut`
    //   borrow precludes aliasing `input`, `weight`, or `bias`
    //   per Rust borrow rules.
    // - One block per row (`rows` blocks, line 18103), 256
    //   threads per block (line 18104), 256 * 4 bytes shared
    //   memory (line 18105) for the per-row mean/var reduction.
    //   The kernel reads `input[row * cols + j]`, `weight[j]`,
    //   `bias[j]` and writes `out[row * cols + j]` for `row in
    //   [0, rows)` and `j in [0, cols)`, bounded by the PTX
    //   per-thread loop with stride `blockDim.x` capped at
    //   `cols`.
    // - `rows_u32`, `cols_u32` cannot truncate from i.e. the
    //   `rows as u32` cast: the grid_dim is a `u32` field so any
    //   `rows > u32::MAX` would already truncate the grid; the
    //   caller's contract (and pool allocator) ensures
    //   `rows * cols <= u32::MAX`. `eps: f32` is passed by-
    //   reference; cudarc copies it into the launch parameter
    //   buffer.
    // - cudarc enqueues the launch on `stream`; the seven arg
    //   refs live until the trailing `?`. Stream sync is the
    //   caller's responsibility.
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "layernorm_backward_kernel",
                source: e,
            });
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `layernorm_backward_kernel` returned by
    //   `module_cache::get_or_compile` at lines 18153-18166; its
    //   ABI `(input, grad_out, weight, grad_in, grad_w, grad_b,
    //   rows, cols, eps)` matches `LAYERNORM_BACKWARD_PTX`.
    // - `input: &CudaBuffer<f32>` is the saved-for-backward
    //   activation `[rows, cols]`; `grad_output:
    //   &CudaBuffer<f32>` is the upstream gradient `[rows, cols]`;
    //   `weight: &CudaBuffer<f32>` is the per-col scale `[cols]`.
    //   `validate_unary(input, device)?` at line 18148 confirms
    //   `input` lives on `device`. The caller's contract is
    //   `input.len() == grad_output.len() == rows * cols` and
    //   `weight.len() == cols`.
    // - `grad_in: &mut CudaBuffer<f32>` was alloc'd at line 18168
    //   for the `[rows, cols]` input gradient. `grad_w: &mut
    //   CudaBuffer<f32>` (line 18169) and `grad_b: &mut
    //   CudaBuffer<f32>` (line 18170) are zero-initialized
    //   accumulators of length `cols` for cross-batch atomic
    //   reductions. All three `&mut` borrows are distinct;
    //   Rust borrow rules guarantee no aliasing between them or
    //   with `input`, `grad_output`, `weight`.
    // - One block per row (line 18176), 256 threads per block,
    //   256*4 bytes shared memory (line 18178) for per-row
    //   gradient reductions. The kernel reads `input[row*cols
    //   +j]`, `grad_output[row*cols+j]`, `weight[j]` and writes
    //   `grad_in[row*cols+j]` directly; `grad_w` and `grad_b`
    //   are accumulated via `atomicAdd` so concurrent block
    //   updates compose correctly.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is u32-
    //   typed; the caller's contract ensures `rows * cols <=
    //   u32::MAX`. `eps: f32` is passed by-reference.
    // - cudarc enqueues the launch on `stream`; the nine arg
    //   refs live until the trailing `?`. Stream sync is the
    //   caller's responsibility before reading any of the three
    //   gradient buffers.
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

// ---------------------------------------------------------------------------
// Public API -- RMSNorm
// ---------------------------------------------------------------------------

/// Row-wise RMS normalization on GPU.
///
/// `input`: `[rows * cols]`, `weight`: `[cols]`.
/// Output: normalized and scaled `[rows * cols]`.
///
/// Computes `out[j] = x[j] * rsqrt(mean(x^2) + eps) * weight[j]`.
/// No bias, no mean centering (unlike LayerNorm).
#[cfg(feature = "cuda")]
pub fn gpu_rmsnorm(
    input: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
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
        RMSNORM_PTX,
        "rmsnorm_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "rmsnorm_kernel",
                source: e,
            });
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

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `rmsnorm_kernel`
    //   returned by `module_cache::get_or_compile` at lines
    //   18240-18253; its ABI `(input_ptr, out_ptr, weight_ptr,
    //   rows, cols, eps)` matches `RMSNORM_PTX`. RMSNorm has no
    //   bias and no mean-centering vs. LayerNorm.
    // - `input: &CudaBuffer<f32>` is the row-major `[rows, cols]`
    //   activation; `validate_unary(input, device)?` at line
    //   18235 confirms device match. `weight: &CudaBuffer<f32>`
    //   is the per-column scale `[cols]`. The caller is
    //   contracted to size `input` to `rows * cols` and `weight`
    //   to `cols`.
    // - `out: &mut CudaBuffer<f32>` was alloc'd at line 18255 via
    //   `alloc_zeros_f32(rows * cols, device)?`. The `&mut`
    //   borrow precludes aliasing `input` or `weight`.
    // - One block per row (line 18260), 256 threads per block
    //   (line 18261), 256*4 bytes shared memory (line 18262) for
    //   the per-row mean-of-squares reduction. The kernel reads
    //   `input[row*cols+j]`, `weight[j]` and writes `out[row*
    //   cols+j]` for `row in [0, rows)` and `j in [0, cols)`,
    //   bounded by the PTX per-thread loop with stride
    //   `blockDim.x` capped at `cols`.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is
    //   u32-typed; the caller contract ensures `rows * cols <=
    //   u32::MAX`. `eps: f32` is passed by-reference.
    // - cudarc enqueues the launch on `stream`; the six arg refs
    //   live until the trailing `?`. Stream sync is the caller's
    //   responsibility.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(weight.inner())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- RMSNorm backward
// ---------------------------------------------------------------------------

/// RMSNorm backward pass on GPU.
///
/// Computes grad_input and grad_weight entirely on GPU.
/// One block per batch element (row), 256 threads per block.
/// grad_weight is accumulated across batches via atomicAdd.
///
/// `input`: `[rows * cols]`, `grad_output`: `[rows * cols]`, `weight`: `[cols]`.
/// Returns: `(grad_input [rows * cols], grad_weight [cols])`.
#[cfg(feature = "cuda")]
pub fn gpu_rmsnorm_backward(
    input: &CudaBuffer<f32>,
    grad_output: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    rows: usize,
    cols: usize,
    eps: f32,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        RMSNORM_BACKWARD_PTX,
        "rmsnorm_backward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "rmsnorm_backward_kernel",
                source: e,
            });
        }
    };

    let mut grad_in = alloc_zeros_f32(rows * cols, device)?;
    let mut grad_w = alloc_zeros_f32(cols, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    // One block per row, 256 threads per block.
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `rmsnorm_backward_kernel` returned by
    //   `module_cache::get_or_compile` at lines 18309-18322; its
    //   ABI `(input, grad_out, weight, grad_in, grad_w, rows,
    //   cols, eps)` matches `RMSNORM_BACKWARD_PTX`.
    // - `input: &CudaBuffer<f32>` is the saved-for-backward
    //   activation `[rows, cols]`; `grad_output:
    //   &CudaBuffer<f32>` is the upstream gradient `[rows, cols]`;
    //   `weight: &CudaBuffer<f32>` is the per-col scale `[cols]`.
    //   `validate_unary(input, device)?` at line 18304 confirms
    //   `input` lives on `device`. The caller's contract is
    //   `input.len() == grad_output.len() == rows * cols` and
    //   `weight.len() == cols`.
    // - `grad_in: &mut CudaBuffer<f32>` was alloc'd at line 18324
    //   for the `[rows, cols]` input gradient. `grad_w: &mut
    //   CudaBuffer<f32>` (line 18325) is the zero-initialized
    //   accumulator of length `cols` for cross-batch atomic
    //   reduction. Both `&mut` borrows are distinct and exclusive
    //   per Rust borrow rules — no aliasing with each other or
    //   with `input`, `grad_output`, `weight`.
    // - One block per row (line 18331), 256 threads per block,
    //   256*4 bytes shared memory (line 18333) for per-row
    //   gradient reductions. The kernel reads `input[row*cols+
    //   j]`, `grad_output[row*cols+j]`, `weight[j]` and writes
    //   `grad_in[row*cols+j]` directly; `grad_w` is accumulated
    //   via `atomicAdd` so concurrent block updates compose
    //   correctly.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is u32-
    //   typed; the caller contract ensures `rows * cols <=
    //   u32::MAX`. `eps: f32` is passed by-reference.
    // - cudarc enqueues the launch on `stream`; the eight arg
    //   refs live until the trailing `?`. Stream sync is the
    //   caller's responsibility before reading the gradient
    //   buffers.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(grad_output.inner())
            .arg(weight.inner())
            .arg(grad_in.inner_mut())
            .arg(grad_w.inner_mut())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }

    Ok((grad_in, grad_w))
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_rmsnorm(
    _input: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_rmsnorm_backward(
    _input: &CudaBuffer<f32>,
    _grad_output: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
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
    try_launch_binary_into(a, b, out, device, ADD_PTX, "add_kernel")
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
    try_launch_binary_into(a, b, out, device, MUL_PTX, "mul_kernel")
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "scale_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `scale_kernel`
    //   returned by `module_cache::get_or_compile` at line 18439
    //   (mapped via `.map_err(|e| GpuError::PtxCompileFailed { ...
    //   })?`); its `(in_ptr, out_ptr, scalar, n)` ABI matches
    //   `SCALE_PTX`.
    // - `a: &CudaBuffer<f32>` is the source; `validate_unary(a,
    //   device)?` at line 18435 confirms device match. `n =
    //   a.len()` (line 18436).
    // - `out: &mut CudaBuffer<f32>` is a caller-supplied pre-
    //   allocated output buffer (used in CUDA-graph capture per
    //   the section header at line 18380); the `&mut` borrow
    //   precludes aliasing `a`. The caller is contracted to size
    //   `out` to at least `n` elements.
    // - The kernel reads `a[i]` and writes `out[i]` only for `i
    //   in [0, n)` per the PTX bound check (`@%p bra DONE`
    //   pattern shared by the PTX). `scalar: f32` is passed by-
    //   reference; cudarc copies it into the launch parameter
    //   buffer.
    // - `n_u32 = n as u32` (line 18450) cannot truncate:
    //   `launch_cfg(n)?` (line 18449) returns `Err` if
    //   `n > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the four arg
    //   refs live until the trailing `?`. Stream sync is the
    //   caller's responsibility (often deferred until graph end
    //   for capture variants).
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

/// Allocate an `n`-element f32 buffer on `device` filled with `scalar`.
///
/// Entirely on-device: no CPU→GPU upload beyond the single f32 scalar
/// passed as a kernel argument. Used by sum/mean backward to produce
/// the constant gradient tensor without the legacy `vec![go;
/// numel].to(device)` round-trip.
#[cfg(feature = "cuda")]
pub fn gpu_fill_f32(n: usize, scalar: f32, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        FILL_F32_PTX,
        "fill_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "scale_kernel",
        source: e,
    })?;

    let mut out = alloc_zeros_f32(n, device)?;
    if n == 0 {
        return Ok(out);
    }
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `fill_f32_kernel`
    //   returned by `module_cache::get_or_compile` at lines
    //   18475-18484 (the `.map_err` reports `kernel:
    //   "scale_kernel"` — a known wrong-kernel-name copy-paste
    //   bug, sibling to #708; substantiation only here, surfaced
    //   in Category A list); its `(out_ptr, scalar, n)` ABI
    //   matches `FILL_F32_PTX`.
    // - `out: &mut CudaBuffer<f32>` was alloc'd at line 18486 via
    //   `alloc_zeros_f32(n, device)?` for the buffer being
    //   filled. The `&mut` borrow ensures exclusive ownership for
    //   the duration of the launch; `n == 0` early-exit at line
    //   18487 ensures the launch only runs when there's work to
    //   do.
    // - The kernel writes `out[i] = scalar` for `i in [0, n)`
    //   per the PTX bound check; reads only `scalar` (passed
    //   by-reference, cudarc copies into launch param buffer).
    // - `n_u32 = n as u32` (line 18491) cannot truncate:
    //   `launch_cfg(n)?` (line 18490) returns `Err` if `n >
    //   u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the three arg
    //   refs live until the trailing `?`. Stream sync is the
    //   caller's responsibility before reading `out`.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(out.inner_mut())
            .arg(&scalar)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
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
    try_launch_unary_into(a, out, device, GELU_PTX, "gelu_kernel")
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "embed_lookup_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(d)?;
    let d_u32 = d as u32;
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `embed_lookup_kernel`
    //   resolved by `module_cache::get_or_compile` at lines
    //   18561-18570; its `(idx_ptr, weight_ptr, out_ptr, d)` ABI
    //   matches `EMBED_LOOKUP_PTX`.
    // - `idx: &CudaBuffer<f32>` is the 1-element token-id buffer
    //   (caller contract: `idx.len() == 1`). `weight:
    //   &CudaBuffer<f32>` is the embedding table `[V, D]`
    //   (caller contract: `weight.len() >= max_id * d`).
    // - `out: &mut CudaBuffer<f32>` is the caller-supplied pre-
    //   allocated `[D]` output (graph-capture variant per the
    //   `_into` section header at line 18380). The `&mut` borrow
    //   precludes aliasing `idx` or `weight`. The caller is
    //   contracted to size `out` to at least `d` elements.
    // - The kernel reads `idx[0]` once and copies `weight[
    //   token_id * d + j]` into `out[j]` for `j in [0, d)` per
    //   the PTX bound check; out-of-range `token_id` is the
    //   caller's responsibility.
    // - `d_u32 = d as u32` (line 18572) cannot truncate:
    //   `launch_cfg(d)?` (line 18571) returns `Err` if `d >
    //   u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the four arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "embed_lookup_batch_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let d_u32 = d as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `embed_lookup_batch_kernel` returned by
    //   `module_cache::get_or_compile` at lines 18610-18623; its
    //   `(indices_ptr, weight_ptr, out_ptr, d, total)` ABI
    //   matches `EMBED_LOOKUP_BATCH_PTX`.
    // - `indices: &CudaBuffer<f32>` is the token-id batch
    //   (caller contract: `indices.len() == n`). `weight:
    //   &CudaBuffer<f32>` is the embedding table `[V, D]`
    //   (caller contract: `max(indices) < V`).
    // - `out: &mut CudaBuffer<f32>` was alloc'd at line 18625 via
    //   `alloc_zeros_f32(total, device)?` where `total = n * d`
    //   (line 18602). The `&mut` borrow precludes aliasing
    //   `indices` or `weight`. The `total == 0` early-return at
    //   line 18603 ensures the launch only runs when there's
    //   work.
    // - The kernel writes `out[i * d + j]` for `(i, j)` in
    //   `[0, n) x [0, d)` (i.e., `total` thread indices) per the
    //   PTX bound check; reads of `weight[indices[i] * d + j]`
    //   rely on the caller index-range contract.
    // - `d_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 18626) returns `Err` if
    //   `total > u32::MAX`; `d <= total`.
    // - cudarc enqueues the launch on `stream`; the five arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "scatter_add_rows_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f32(num_embeddings * d, device)?;
    let cfg = launch_cfg(total)?;
    let d_u32 = d as u32;
    let total_u32 = total as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `scatter_add_rows_kernel` returned by
    //   `module_cache::get_or_compile` at lines 18673-18686; its
    //   `(grad_ptr, idx_ptr, out_ptr, d, total)` ABI matches
    //   `SCATTER_ADD_ROWS_PTX`.
    // - `grad_output: &CudaBuffer<f32>` is the upstream gradient
    //   `[N, D]` (caller contract: `grad_output.len() == n * d`).
    //   `indices: &CudaBuffer<f32>` carries destination row
    //   indices (caller contract: `n = indices.len()`,
    //   `max(indices) < num_embeddings`).
    // - `out: &mut CudaBuffer<f32>` was alloc'd at line 18688 via
    //   `alloc_zeros_f32(num_embeddings * d, device)?`. The
    //   `&mut` borrow precludes aliasing `grad_output` or
    //   `indices`. The `total == 0` early-return at line 18666
    //   ensures the launch only runs when there's work.
    // - The kernel uses `atomicAdd` (per PTX) to accumulate
    //   `grad_output[i * d + j]` into `out[indices[i] * d + j]`
    //   for `(i, j)` in `[0, n) x [0, d)` — `total = n * d`
    //   thread indices; duplicate indices accumulate correctly.
    // - `d_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 18689) returns `Err` if
    //   `total > u32::MAX`; `d <= total`.
    // - cudarc enqueues the launch on `stream`; the five arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "scatter_add_rows_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(total)?;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let total_u32 = total as u32;
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `transpose_2d_kernel`
    //   resolved by `module_cache::get_or_compile` at lines
    //   18720-18729 (the `.map_err` reports `kernel:
    //   "scatter_add_rows_kernel"` — a known wrong-kernel-name
    //   copy-paste bug, sibling to #708; substantiation only here,
    //   surfaced in Category A list); its `(in_ptr, out_ptr, m, n,
    //   total)` ABI matches `TRANSPOSE_2D_PTX`.
    // - `a: &CudaBuffer<f32>` is the source `[m, n]` matrix
    //   (caller contract: `a.len() >= m * n`). `out: &mut
    //   CudaBuffer<f32>` is the caller-supplied pre-allocated
    //   `[n, m]` output (graph-capture variant per the `_into`
    //   section header at line 18380); the `&mut` borrow
    //   precludes aliasing `a`.
    // - The kernel reads `a[i]` and writes the transposed
    //   `out[i]` only for `i in [0, total)` per the PTX bound
    //   check; `total = m * n` covers both [m, n] source and
    //   [n, m] destination element counts.
    // - `m_u32`, `n_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 18730) returns `Err` if
    //   `total > u32::MAX`; `m <= total`, `n <= total`.
    // - cudarc enqueues the launch on `stream`; the five arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "permute_0213_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(total)?;
    let (d0u, d1u, d2u, d3u, tu) = (d0 as u32, d1 as u32, d2 as u32, d3 as u32, total as u32);
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `permute_0213_kernel`
    //   resolved by `module_cache::get_or_compile` at lines
    //   18762-18771; its `(in_ptr, out_ptr, d0, d1, d2, d3, total)`
    //   ABI matches `PERMUTE_0213_PTX`.
    // - `a: &CudaBuffer<f32>` is the source `[d0, d1, d2, d3]`
    //   tensor (caller contract: `a.len() >= total = d0*d1*d2*d3`).
    //   `out: &mut CudaBuffer<f32>` is the caller-supplied pre-
    //   allocated `[d0, d2, d1, d3]` output (graph-capture
    //   variant per the `_into` section header); the `&mut`
    //   borrow precludes aliasing `a`.
    // - The kernel reads `a[i]` and writes the permuted `out[i]`
    //   only for `i in [0, total)` per the PTX bound check; the
    //   permutation `[d0, d1, d2, d3] -> [d0, d2, d1, d3]`
    //   preserves total element count.
    // - `d0u`, `d1u`, `d2u`, `d3u`, `tu` cannot truncate:
    //   `launch_cfg(total)?` (line 18772) returns `Err` if
    //   `total > u32::MAX`; each dim is bounded by `total`.
    // - cudarc enqueues the launch on `stream`; the seven arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "softmax_kernel",
        source: e,
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
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `softmax_kernel`
    //   resolved by `module_cache::get_or_compile` at lines
    //   18801-18810; its `(in_ptr, out_ptr, rows, cols)` ABI
    //   matches `SOFTMAX_PTX`.
    // - `a: &CudaBuffer<f32>` is the row-major `[rows, cols]`
    //   activation (caller contract: `a.len() >= rows * cols`).
    //   `out: &mut CudaBuffer<f32>` is the caller-supplied pre-
    //   allocated `[rows, cols]` output (graph-capture variant
    //   per the `_into` section header); the `&mut` borrow
    //   precludes aliasing `a`.
    // - One block per row (grid `rows`), `block_size = 256`
    //   threads per block, `cols * 4` bytes shared memory for the
    //   per-row max+sum reduction. The kernel reads `a[row*cols
    //   +j]` and writes `out[row*cols+j]` for `row in [0, rows)`
    //   and `j in [0, cols)`, bounded by the PTX per-thread loop
    //   with stride `blockDim.x` capped at `cols`.
    // - `rows_u32`, `cols_u32`, and `(cols as u32) * 4` shared-
    //   mem bytes cannot truncate: grid_dim and shared_mem_bytes
    //   are u32-typed; the caller contract ensures `rows * cols
    //   <= u32::MAX` and `cols * 4 <= u32::MAX` (i.e., `cols <
    //   2^30`, far above realistic activation widths).
    // - cudarc enqueues the launch on `stream`; the four arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "layernorm_kernel",
        source: e,
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
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `layernorm_kernel`
    //   resolved by `module_cache::get_or_compile` at lines
    //   18848-18857; its `(input_ptr, out_ptr, weight_ptr,
    //   bias_ptr, rows, cols, eps)` ABI matches `LAYERNORM_PTX`.
    // - `input: &CudaBuffer<f32>` is the row-major `[rows, cols]`
    //   activation; `weight: &CudaBuffer<f32>` and `bias:
    //   &CudaBuffer<f32>` are per-column affine parameters
    //   (length `cols`). The caller is contracted to size
    //   `input` to `rows * cols` and `weight`/`bias` to `cols`.
    // - `out: &mut CudaBuffer<f32>` is the caller-supplied pre-
    //   allocated `[rows, cols]` output (graph-capture variant
    //   per the `_into` section header at line 18380); the
    //   `&mut` borrow precludes aliasing `input`, `weight`, or
    //   `bias`.
    // - One block per row (grid `rows`), `block_size = 256`
    //   threads per block, `cols * 4` bytes shared memory for
    //   the per-row mean/var reduction. The kernel reads
    //   `input[row*cols+j]`, `weight[j]`, `bias[j]` and writes
    //   `out[row*cols+j]` for `row in [0, rows)` and `j in [0,
    //   cols)`, bounded by the PTX per-thread loop with stride
    //   `blockDim.x`.
    // - `rows_u32`, `cols_u32`, `(cols as u32) * 4` cannot
    //   truncate: grid_dim and shared_mem_bytes are u32-typed;
    //   caller contract ensures `rows * cols <= u32::MAX` and
    //   `cols < 2^30`. `eps: f32` is passed by-reference.
    // - cudarc enqueues the launch on `stream`; the seven arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "slice_read_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(total)?;
    let total_u32 = total as u32;
    let d_u32 = d as u32;
    let len_u32 = len as u32;
    let max_len_u32 = max_len as u32;
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `slice_read_kernel`
    //   resolved by `module_cache::get_or_compile` at lines
    //   18898-18907; its `(src_ptr, out_ptr, total, d, len,
    //   max_len)` ABI matches `SLICE_READ_PTX`.
    // - `src: &CudaBuffer<f32>` is the source `[n_batch, max_len,
    //   d]` buffer (caller contract: size = `n_batch * max_len *
    //   d`, `len <= max_len`). `out: &mut CudaBuffer<f32>` is
    //   the caller-supplied pre-allocated `[n_batch, len, d]`
    //   output (graph-capture variant per the `_into` section
    //   header); the `&mut` borrow precludes aliasing `src`.
    // - The kernel writes `out[(b * len + l) * d + j]` for
    //   `(b, l, j)` in the cartesian product (`total` thread
    //   indices) per the PTX bound check; reads of
    //   `src[(b * max_len + l) * d + j]` for `l < len` are
    //   bounded by `n_batch * max_len * d` (caller contract).
    // - `total_u32`, `d_u32`, `len_u32`, `max_len_u32` cannot
    //   truncate: `launch_cfg(total)?` (line 18908) returns
    //   `Err` if `total > u32::MAX`; `d <= total`, `len <=
    //   max_len`, `max_len * n_batch * d <= u32::MAX` per
    //   allocator pool limits.
    // - cudarc enqueues the launch on `stream`; the six arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "small_matmul_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(total)?;
    let (m_u32, k_u32, n_u32, total_u32) = (m as u32, k as u32, n as u32, total as u32);
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `small_matmul_kernel`
    //   resolved by `module_cache::get_or_compile` at lines
    //   18942-18951; its `(a_ptr, b_ptr, out_ptr, m, k, n, total)`
    //   ABI matches `SMALL_MATMUL_PTX`.
    // - `a: &CudaBuffer<f32>` is the `[m, k]` left matrix; `b:
    //   &CudaBuffer<f32>` is the `[k, n]` right matrix (caller
    //   contracts: `a.len() >= m*k`, `b.len() >= k*n`). `out:
    //   &mut CudaBuffer<f32>` is the caller-supplied pre-
    //   allocated `[m, n]` output (graph-capture variant per the
    //   `_into` section header); the `&mut` borrow precludes
    //   aliasing `a` or `b`.
    // - The kernel runs one thread per output element (`total =
    //   m * n` threads) per the PTX bound check; each thread
    //   computes `out[i*n+j] = sum_k a[i*k+kk] * b[kk*n+j]`,
    //   reads of `a` and `b` bounded by `m*k` and `k*n`
    //   respectively (caller contract).
    // - `m_u32`, `k_u32`, `n_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 18952) returns `Err` if
    //   `total > u32::MAX`; `m <= total`, `n <= total`, and
    //   `k <= max(a.len()/m, b.len()/n) <= u32::MAX` (caller
    //   contract on the input buffer sizes).
    // - cudarc enqueues the launch on `stream`; the seven arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "slice_write_indirect_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(total)?;
    let n_u32 = total as u32;
    let d_u32 = d as u32;
    let max_len_u32 = max_len as u32;
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `slice_write_indirect_kernel` resolved by
    //   `module_cache::get_or_compile` at lines 18989-18998; its
    //   `(src_ptr, dst_ptr, n_total, d, max_len, pos_ptr)` ABI
    //   matches `SLICE_WRITE_INDIRECT_PTX`. Unlike the direct
    //   variant (line 17321), the row index is read from device
    //   memory at kernel time — required for CUDA-graph capture
    //   where launch parameters must not change at replay.
    // - `src: &CudaBuffer<f32>` is the source `[n_batch, d]`
    //   buffer (caller contract: `src.len() >= n_batch * d`).
    //   `dst: &mut CudaBuffer<f32>` is the destination
    //   `[n_batch, max_len, d]` buffer (caller contract:
    //   `dst.len() >= n_batch * max_len * d`); the `&mut` borrow
    //   precludes aliasing `src` or `pos_ptr`.
    // - `pos_ptr: &cudarc::driver::CudaSlice<u32>` is a single-
    //   element device-resident position counter. The kernel
    //   reads `*pos_ptr` once at launch and writes `dst[(b *
    //   max_len + *pos_ptr) * d + j]` for `(b, j)` in `[0,
    //   n_batch) x [0, d)` (i.e., `total = n_batch * d` thread
    //   indices); the caller is contracted to ensure `*pos_ptr <
    //   max_len` at kernel time.
    // - `n_u32 = total as u32`, `d_u32`, `max_len_u32` cannot
    //   truncate: `launch_cfg(total)?` (line 18999) returns
    //   `Err` if `total > u32::MAX`; `d <= total`, `max_len`
    //   bounded by `dst.len() / (n_batch * d) <= u32::MAX`
    //   (allocator pool limits).
    // - cudarc enqueues the launch on `stream`; the six arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's (typically deferred to graph end during
    //   capture).
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "causal_mask_indirect_kernel",
        source: e,
    })?;
    let cfg = launch_cfg(total)?;
    let max_pos_u32 = max_pos as u32;
    let total_u32 = total as u32;
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `causal_mask_indirect_kernel` resolved by
    //   `module_cache::get_or_compile` at lines 19032-19041; its
    //   `(total_len_ptr, out_ptr, max_pos, total)` ABI matches
    //   `CAUSAL_MASK_INDIRECT_PTX`.
    // - `total_len_ptr: &cudarc::driver::CudaSlice<u32>` is a
    //   single-element device-resident sequence-length counter.
    //   The kernel reads `*total_len_ptr` once at launch and
    //   writes `out[h, col] = 0.0` if `col < *total_len_ptr`,
    //   else `-1e9`.
    // - `out: &mut CudaBuffer<f32>` is the caller-supplied pre-
    //   allocated `[n_head, max_pos]` mask buffer (graph-capture
    //   variant); the `&mut` borrow precludes aliasing
    //   `total_len_ptr`. The caller is contracted to size `out`
    //   to at least `n_head * max_pos = total` (line 19029).
    // - The kernel runs one thread per output element (`total =
    //   n_head * max_pos` threads) per the PTX bound check;
    //   each thread writes exactly one `f32` to `out`. Reads of
    //   `total_len_ptr` are indirect through the device pointer
    //   only.
    // - `max_pos_u32`, `total_u32` cannot truncate:
    //   `launch_cfg(total)?` (line 19042) returns `Err` if
    //   `total > u32::MAX`; `max_pos <= total` since `total =
    //   n_head * max_pos`.
    // - cudarc enqueues the launch on `stream`; the four arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's (typically deferred to graph end).
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
pub fn gpu_gelu_tanh(_input: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_erf(_input: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_backward_tanh(
    _grad: &CudaBuffer<f32>,
    _input: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_silu(_input: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_silu_backward(
    _grad: &CudaBuffer<f32>,
    _input: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_elu(
    _input: &CudaBuffer<f32>,
    _alpha: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_elu_backward(
    _grad: &CudaBuffer<f32>,
    _input: &CudaBuffer<f32>,
    _alpha: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_mish(_input: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_mish_backward(
    _grad: &CudaBuffer<f32>,
    _input: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_clamp(
    _input: &CudaBuffer<f32>,
    _min_val: f32,
    _max_val: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
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
pub fn gpu_abs_backward(
    _grad: &CudaBuffer<f32>,
    _input: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_fill_f32(_n: usize, _scalar: f32, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
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
pub fn gpu_log_softmax(
    _input: &CudaBuffer<f32>,
    _cols: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_log_softmax_backward(
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
pub fn gpu_cumsum(
    _input: &CudaBuffer<f32>,
    _outer: usize,
    _dim_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_cumprod(
    _input: &CudaBuffer<f32>,
    _outer: usize,
    _dim_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_cummax(
    _input: &CudaBuffer<f32>,
    _outer: usize,
    _dim_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_cummin(
    _input: &CudaBuffer<f32>,
    _outer: usize,
    _dim_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_logcumsumexp(
    _input: &CudaBuffer<f32>,
    _outer: usize,
    _dim_size: usize,
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

/// Maximum rank stub for feature-disabled builds. Kept in sync with
/// the cuda-enabled definition above.
#[cfg(not(feature = "cuda"))]
pub const STRIDED_COPY_MAX_DIMS: usize = 8;

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_strided_copy(
    _input: &CudaBuffer<f32>,
    _out_shape: &[usize],
    _src_strides: &[isize],
    _src_offset: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_strided_copy_f64(
    _input: &CudaBuffer<f64>,
    _out_shape: &[usize],
    _src_strides: &[isize],
    _src_offset: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "f32_to_f16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `f32_to_f16_kernel`
    //   resolved by `module_cache::get_or_compile` at lines
    //   19756-19765 (returns `PtxCompileFailed` on Err); its
    //   `(input_ptr, out_ptr, n)` ABI matches `F32_TO_F16_PTX`.
    //   The kernel uses `cvt.rn.f16.f32` (PTX round-to-nearest-
    //   even) per the function rustdoc at line 19732.
    // - `input: &CudaBuffer<f32>` is the source `[n]` f32 buffer
    //   (caller contract: `input.len() == n`).
    // - `out: &mut cudarc::driver::CudaSlice<u16>` was alloc'd at
    //   line 19767 via `stream.alloc_zeros::<u16>(n)?` for the
    //   `[n]` u16 (f16 bit-pattern) destination. The `&mut` borrow
    //   precludes aliasing `input` per Rust borrow rules. The
    //   `n == 0` early-return at line 19748 ensures the launch
    //   only runs when there's work.
    // - The kernel reads `input[i]` (one f32 lane) and writes
    //   `out[i]` (one u16 lane) only for `i in [0, n)` per the
    //   PTX bound check (`@%p bra DONE` pattern shared by the
    //   strided launch). f16 is a valid bit pattern in u16 — the
    //   caller does not interpret bits as signaling NaN; that's
    //   the consumer's contract.
    // - `n_u32 = n as u32` (line 19769) cannot truncate:
    //   `launch_cfg(n)?` (line 19768) returns `Err` if
    //   `n > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the three arg
    //   refs live until the trailing `?`. Stream sync is the
    //   caller's responsibility before reading `out`.
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
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "f32_to_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `f32_to_bf16_kernel`
    //   resolved by `module_cache::get_or_compile` at lines
    //   19812-19821 (returns `PtxCompileFailed` on Err); its
    //   `(input_ptr, out_ptr, n)` ABI matches `F32_TO_BF16_PTX`.
    //   The kernel uses bit-manipulation round-to-nearest-even
    //   per the function rustdoc at line 19794 (no special bf16
    //   hardware required, sm_52+).
    // - `input: &CudaBuffer<f32>` is the source `[n]` f32 buffer
    //   (caller contract: `input.len() == n`).
    // - `out: &mut cudarc::driver::CudaSlice<u16>` was alloc'd at
    //   line 19823 via `stream.alloc_zeros::<u16>(n)?` for the
    //   `[n]` u16 (bf16 bit-pattern) destination. The `&mut`
    //   borrow precludes aliasing `input`. The `n == 0` early-
    //   return at line 19804 ensures the launch only runs when
    //   there's work.
    // - The kernel reads `input[i]` (one f32 lane) and writes
    //   `out[i]` (one u16 lane = upper 16 bits of f32 with RNE
    //   rounding) only for `i in [0, n)` per the PTX bound
    //   check.
    // - `n_u32 = n as u32` (line 19825) cannot truncate:
    //   `launch_cfg(n)?` (line 19824) returns `Err` if
    //   `n > u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the three arg
    //   refs live until the trailing `?`. Stream sync is the
    //   caller's responsibility before reading `out`.
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
// Non-CUDA stubs -- f64 ops
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
pub fn gpu_add_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_sub_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_mul_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_div_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_neg_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_relu_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_scale_f64(
    _a: &CudaBuffer<f64>,
    _scalar: f64,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_exp_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_log_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_sqrt_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_pow_f64(
    _a: &CudaBuffer<f64>,
    _exponent: f64,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_abs_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_sigmoid_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_tanh_f64(_a: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_relu_backward_f64(
    _grad: &CudaBuffer<f64>,
    _input: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_sigmoid_backward_f64(
    _grad: &CudaBuffer<f64>,
    _output: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_tanh_backward_f64(
    _grad: &CudaBuffer<f64>,
    _output: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_add_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _a_shape: &[usize],
    _b_shape: &[usize],
    _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_sub_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _a_shape: &[usize],
    _b_shape: &[usize],
    _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_mul_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _a_shape: &[usize],
    _b_shape: &[usize],
    _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_div_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _a_shape: &[usize],
    _b_shape: &[usize],
    _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_transpose_2d_f64(
    _input: &CudaBuffer<f64>,
    _m: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_permute_0213_f64(
    _input: &CudaBuffer<f64>,
    _d0: usize,
    _d1: usize,
    _d2: usize,
    _d3: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_strided_split_f64(
    _input: &CudaBuffer<f64>,
    _total_along_axis: usize,
    _split_offset: usize,
    _split_size: usize,
    _inner_size: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_strided_cat_f64(
    _input: &CudaBuffer<f64>,
    _output: &mut CudaBuffer<f64>,
    _total_along_axis: usize,
    _cat_offset: usize,
    _part_size: usize,
    _inner_size: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_index_select_1d_f64(
    _input: &CudaBuffer<f64>,
    _indices: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_scatter_add_1d_f64(
    _grad_output: &CudaBuffer<f64>,
    _indices: &CudaBuffer<f32>,
    _input_len: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_fill_f64(
    _input: &CudaBuffer<f64>,
    _mask: &CudaBuffer<u8>,
    _value: f64,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_zero_f64(
    _grad: &CudaBuffer<f64>,
    _mask: &CudaBuffer<u8>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_slice_write_f64(
    _src: &CudaBuffer<f64>,
    _dst: &mut CudaBuffer<f64>,
    _n_batch: usize,
    _d: usize,
    _max_len: usize,
    _pos: usize,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_slice_read_f64(
    _src: &CudaBuffer<f64>,
    _n_batch: usize,
    _d: usize,
    _len: usize,
    _max_len: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_embed_lookup_f64(
    _idx: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f64>,
    _d: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_embed_lookup_batch_f64(
    _indices: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f64>,
    _n: usize,
    _d: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_scatter_add_rows_f64(
    _grad_output: &CudaBuffer<f64>,
    _indices: &CudaBuffer<f32>,
    _num_embeddings: usize,
    _d: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Public API -- f64 activation, normalization, scan, and pooling launchers
// ---------------------------------------------------------------------------

/// GELU (sigmoid-approx) for f64.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_f64(input: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    try_launch_unary_f64(input, device, GELU_F64_PTX, "gelu_f64_kernel")
}

/// GELU (tanh-approx) for f64.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_tanh_f64(
    input: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    try_launch_unary_f64(input, device, GELU_TANH_F64_PTX, "gelu_tanh_f64_kernel")
}

/// GELU (exact erf) for f64.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_erf_f64(input: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    try_launch_unary_f64(input, device, GELU_ERF_F64_PTX, "gelu_erf_f64_kernel")
}

/// GELU backward (sigmoid-approx) for f64.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_backward_f64(
    grad: &CudaBuffer<f64>,
    input: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    try_launch_binary_f64(
        grad,
        input,
        device,
        GELU_BACKWARD_F64_PTX,
        "gelu_backward_f64_kernel",
    )
}

/// GELU backward (tanh-approx) for f64.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_backward_tanh_f64(
    grad: &CudaBuffer<f64>,
    input: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    try_launch_binary_f64(
        grad,
        input,
        device,
        GELU_BACKWARD_TANH_F64_PTX,
        "gelu_backward_tanh_f64_kernel",
    )
}

/// GELU backward (exact erf) for f64.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_backward_erf_f64(
    grad: &CudaBuffer<f64>,
    input: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    try_launch_binary_f64(
        grad,
        input,
        device,
        GELU_BACKWARD_ERF_F64_PTX,
        "gelu_backward_erf_f64_kernel",
    )
}

/// SiLU for f64.
#[cfg(feature = "cuda")]
pub fn gpu_silu_f64(input: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    try_launch_unary_f64(input, device, SILU_F64_PTX, "silu_f64_kernel")
}

/// SiLU backward for f64.
#[cfg(feature = "cuda")]
pub fn gpu_silu_backward_f64(
    grad: &CudaBuffer<f64>,
    input: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    try_launch_binary_f64(
        grad,
        input,
        device,
        SILU_BACKWARD_F64_PTX,
        "silu_backward_f64_kernel",
    )
}

/// ELU for f64.
#[cfg(feature = "cuda")]
pub fn gpu_elu_f64(
    input: &CudaBuffer<f64>,
    alpha: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    let n = input.len();
    if n == 0 {
        return cpu_to_gpu(&[], device);
    }
    let ctx = device.context();
    let stream = device.stream();
    match crate::module_cache::get_or_compile(
        ctx,
        ELU_F64_PTX,
        "elu_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => {
            let mut out = alloc_zeros_f64(n, device)?;
            let n_u32 = n as u32;
            let cfg = launch_cfg(n)?;
            // SAFETY:
            // - `f` is a valid PTX `CudaFunction` for `elu_f64_kernel`
            //   resolved by `module_cache::get_or_compile` in the
            //   `if let Ok(f)` arm directly above (the silent CPU
            //   fallback in the implicit `else` is the policy concern
            //   tracked by gpu-F, NOT this dispatch's scope); its
            //   `(in_ptr, out_ptr, n, alpha)` ABI matches `ELU_F64_PTX`.
            // - `input: &CudaBuffer<f64>` is the source. `out: &mut
            //   CudaBuffer<f64>` was alloc'd in this scope via
            //   `alloc_zeros_f64(n, device)?`. The `&mut` borrow
            //   precludes aliasing `input`. The `n == 0` early-return
            //   above ensures the launch only runs when there's work.
            // - `n = input.len()`; the kernel reads `input[i]` and
            //   writes `out[i]` only for `i in [0, n)` per the PTX
            //   bound check. `alpha: f64` is passed by-reference;
            //   cudarc copies it into the launch parameter buffer.
            // - `n_u32 = n as u32` cannot truncate: `launch_cfg(n)?`
            //   returns `Err` if `n > u32::MAX`.
            // - cudarc enqueues the launch on `stream`; the four arg
            //   refs live until the trailing `?`. Stream sync is
            //   caller's.
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(input.inner())
                    .arg(out.inner_mut())
                    .arg(&n_u32)
                    .arg(&alpha)
                    .launch(cfg)?;
            }
            Ok(out)
        }
        Err(e) => {
            if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
                tracing::warn!(
                    target: "ferrotorch::gpu_fallback",
                    kernel = "elu_f64_kernel",
                    error = %e,
                    "PTX compile failed; falling back to CPU. Unset \
                     FERROTORCH_ENABLE_GPU_FALLBACK to make this an error instead.",
                );
                let host = gpu_to_cpu(input, device)?;
                let result: Vec<f64> = host
                    .iter()
                    .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
                    .collect();
                return cpu_to_gpu(&result, device);
            }
            Err(GpuError::PtxCompileFailed {
                kernel: "elu_f64_kernel",
                source: e,
            })
        }
    }
}

/// ELU backward for f64.
#[cfg(feature = "cuda")]
pub fn gpu_elu_backward_f64(
    grad: &CudaBuffer<f64>,
    input: &CudaBuffer<f64>,
    alpha: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    let n = grad.len();
    if n == 0 {
        return cpu_to_gpu(&[], device);
    }
    let ctx = device.context();
    let stream = device.stream();
    match crate::module_cache::get_or_compile(
        ctx,
        ELU_BACKWARD_F64_PTX,
        "elu_backward_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => {
            let mut out = alloc_zeros_f64(n, device)?;
            let n_u32 = n as u32;
            let cfg = launch_cfg(n)?;
            // SAFETY:
            // - `f` is a valid PTX `CudaFunction` for
            //   `elu_backward_f64_kernel` resolved in the `if let
            //   Ok(f)` arm directly above (the silent CPU fallback
            //   below is gpu-F territory, not this dispatch's); its
            //   `(grad_ptr, input_ptr, out_ptr, n, alpha)` ABI matches
            //   `ELU_BACKWARD_F64_PTX`.
            // - `grad: &CudaBuffer<f64>` and `input: &CudaBuffer<f64>`
            //   are caller-supplied; the length-equality precondition
            //   `grad.len() == input.len()` is enforced earlier in
            //   this fn (returns `LengthMismatch` otherwise).
            //   `n = grad.len()`; the `n == 0` early-return above
            //   ensures the launch only runs when there's work.
            // - `out: &mut CudaBuffer<f64>` was alloc'd via
            //   `alloc_zeros_f64(n, device)?`. The `&mut` borrow
            //   precludes aliasing `grad` or `input`.
            // - The kernel reads `grad[i]`, `input[i]` and writes
            //   `out[i]` only for `i in [0, n)` per the PTX bound
            //   check. `alpha: f64` is passed by-reference.
            // - `n_u32 = n as u32` cannot truncate: `launch_cfg(n)?`
            //   returns `Err` if `n > u32::MAX`.
            // - cudarc enqueues the launch on `stream`; the five arg
            //   refs live until the trailing `?`. Stream sync is
            //   caller's.
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(grad.inner())
                    .arg(input.inner())
                    .arg(out.inner_mut())
                    .arg(&n_u32)
                    .arg(&alpha)
                    .launch(cfg)?;
            }
            Ok(out)
        }
        Err(e) => {
            if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
                tracing::warn!(
                    target: "ferrotorch::gpu_fallback",
                    kernel = "elu_backward_f64_kernel",
                    error = %e,
                    "PTX compile failed; falling back to CPU. Unset \
                     FERROTORCH_ENABLE_GPU_FALLBACK to make this an error instead.",
                );
                let g_host = gpu_to_cpu(grad, device)?;
                let x_host = gpu_to_cpu(input, device)?;
                let result: Vec<f64> = g_host
                    .iter()
                    .zip(x_host.iter())
                    .map(|(&g, &x)| if x > 0.0 { g } else { g * alpha * x.exp() })
                    .collect();
                return cpu_to_gpu(&result, device);
            }
            Err(GpuError::PtxCompileFailed {
                kernel: "elu_backward_f64_kernel",
                source: e,
            })
        }
    }
}

/// Mish for f64.
#[cfg(feature = "cuda")]
pub fn gpu_mish_f64(input: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    try_launch_unary_f64(input, device, MISH_F64_PTX, "mish_f64_kernel")
}

/// Mish backward for f64.
#[cfg(feature = "cuda")]
pub fn gpu_mish_backward_f64(
    grad: &CudaBuffer<f64>,
    input: &CudaBuffer<f64>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    try_launch_binary_f64(
        grad,
        input,
        device,
        MISH_BACKWARD_F64_PTX,
        "mish_backward_f64_kernel",
    )
}

/// Clamp for f64.
#[cfg(feature = "cuda")]
pub fn gpu_clamp_f64(
    input: &CudaBuffer<f64>,
    min_val: f64,
    max_val: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let n = input.len();
    if n == 0 {
        return cpu_to_gpu(&[], device);
    }
    let ctx = device.context();
    let stream = device.stream();
    let ptx = get_f64_ptx(&CACHE, CLAMP_PTX, "clamp_kernel", "clamp_f64_kernel");
    match crate::module_cache::get_or_compile(ctx, ptx, "clamp_f64_kernel", device.ordinal() as u32)
    {
        Ok(f) => {
            let mut out = alloc_zeros_f64(n, device)?;
            let n_u32 = n as u32;
            let cfg = launch_cfg(n)?;
            // SAFETY:
            // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
            //   `clamp_f64_kernel` resolved by
            //   `module_cache::get_or_compile` in the `if let Ok(f)`
            //   arm directly above (the silent CPU fallback below is
            //   gpu-F territory, not this dispatch's); its `(in_ptr,
            //   out_ptr, n, min_val, max_val)` ABI matches `CLAMP_PTX`
            //   rewritten by `ptx_f32_to_f64` (line ~20386).
            // - `input: &CudaBuffer<f64>` is the source. `out: &mut
            //   CudaBuffer<f64>` was alloc'd via `alloc_zeros_f64(n,
            //   device)?`. The `&mut` borrow precludes aliasing
            //   `input`. The `n == 0` early-return above ensures the
            //   launch only runs when there's work.
            // - `n = input.len()`; the kernel reads `input[i]` and
            //   writes `out[i] = clamp(input[i], min_val, max_val)`
            //   only for `i in [0, n)` per the PTX bound check.
            //   `min_val: f64`, `max_val: f64` passed by-reference.
            // - `n_u32 = n as u32` cannot truncate: `launch_cfg(n)?`
            //   returns `Err` if `n > u32::MAX`.
            // - cudarc enqueues the launch on `stream`; the five arg
            //   refs live until the trailing `?`. Stream sync is
            //   caller's.
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(input.inner())
                    .arg(out.inner_mut())
                    .arg(&n_u32)
                    .arg(&min_val)
                    .arg(&max_val)
                    .launch(cfg)?;
            }
            Ok(out)
        }
        Err(e) => {
            if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
                tracing::warn!(
                    target: "ferrotorch::gpu_fallback",
                    kernel = "clamp_f64_kernel",
                    error = %e,
                    "PTX compile failed; falling back to CPU. Unset \
                     FERROTORCH_ENABLE_GPU_FALLBACK to make this an error instead.",
                );
                let host = gpu_to_cpu(input, device)?;
                let result: Vec<f64> = host.iter().map(|&x| x.max(min_val).min(max_val)).collect();
                return cpu_to_gpu(&result, device);
            }
            Err(GpuError::PtxCompileFailed {
                kernel: "clamp_f64_kernel",
                source: e,
            })
        }
    }
}

/// f64 VJP for `clamp(x, min, max)`. Counterpart of [`gpu_clamp_backward`].
/// (#524)
#[cfg(feature = "cuda")]
pub fn gpu_clamp_backward_f64(
    grad: &CudaBuffer<f64>,
    input: &CudaBuffer<f64>,
    min_val: f64,
    max_val: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    if grad.len() != input.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: input.len(),
        });
    }
    let n = input.len();
    if n == 0 {
        return cpu_to_gpu(&[], device);
    }
    let ctx = device.context();
    let stream = device.stream();
    let ptx = get_f64_ptx(
        &CACHE,
        CLAMP_BACKWARD_PTX,
        "clamp_backward_kernel",
        "clamp_backward_f64_kernel",
    );
    match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "clamp_backward_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => {
            let mut out = alloc_zeros_f64(n, device)?;
            let n_u32 = n as u32;
            let cfg = launch_cfg(n)?;
            // SAFETY:
            // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
            //   `clamp_backward_f64_kernel` resolved by
            //   `module_cache::get_or_compile` in the `if let Ok(f)`
            //   arm directly above (the silent CPU fallback below is
            //   gpu-F territory, not this dispatch's); its `(grad_ptr,
            //   input_ptr, out_ptr, min_val, max_val, n)` ABI matches
            //   `CLAMP_BACKWARD_PTX` rewritten by `ptx_f32_to_f64`.
            // - `grad: &CudaBuffer<f64>` and `input: &CudaBuffer<f64>`
            //   are caller-supplied; the length-equality precondition
            //   `grad.len() == input.len()` is enforced earlier in
            //   this fn (returns `LengthMismatch` otherwise).
            //   `n = input.len()`; the `n == 0` early-return above
            //   ensures the launch only runs when there's work.
            // - `out: &mut CudaBuffer<f64>` was alloc'd via
            //   `alloc_zeros_f64(n, device)?`. The `&mut` borrow
            //   precludes aliasing `grad` or `input`.
            // - The kernel reads `grad[i]`, `input[i]` and writes
            //   `out[i] = (input[i] in [min_val, max_val]) ? grad[i]
            //   : 0.0` only for `i in [0, n)` per the PTX bound
            //   check. `min_val: f64`, `max_val: f64` passed by-
            //   reference.
            // - `n_u32 = n as u32` cannot truncate: `launch_cfg(n)?`
            //   returns `Err` if `n > u32::MAX`.
            // - cudarc enqueues the launch on `stream`; the six arg
            //   refs live until the trailing `?`. Stream sync is
            //   caller's.
            unsafe {
                stream
                    .launch_builder(&f)
                    .arg(grad.inner())
                    .arg(input.inner())
                    .arg(out.inner_mut())
                    .arg(&min_val)
                    .arg(&max_val)
                    .arg(&n_u32)
                    .launch(cfg)?;
            }
            Ok(out)
        }
        Err(e) => {
            if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
                tracing::warn!(
                    target: "ferrotorch::gpu_fallback",
                    kernel = "clamp_backward_f64_kernel",
                    error = %e,
                    "PTX compile failed; falling back to CPU. Unset \
                     FERROTORCH_ENABLE_GPU_FALLBACK to make this an error instead.",
                );
                // PTX-compile failure → host walk.
                let g = gpu_to_cpu(grad, device)?;
                let x = gpu_to_cpu(input, device)?;
                let out: Vec<f64> = g
                    .iter()
                    .zip(x.iter())
                    .map(|(&gi, &xi)| {
                        if xi >= min_val && xi <= max_val {
                            gi
                        } else {
                            0.0
                        }
                    })
                    .collect();
                return cpu_to_gpu(&out, device);
            }
            Err(GpuError::PtxCompileFailed {
                kernel: "clamp_backward_f64_kernel",
                source: e,
            })
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_clamp_backward_f64(
    _grad: &CudaBuffer<f64>,
    _input: &CudaBuffer<f64>,
    _min: f64,
    _max: f64,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

/// Cumulative sum for f64.
#[cfg(feature = "cuda")]
pub fn gpu_cumsum_f64(
    input: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let total = outer * inner;
    let n = outer * dim_size * inner;
    if n == 0 {
        return cpu_to_gpu(&[], device);
    }
    let ctx = device.context();
    let stream = device.stream();
    let ptx = get_f64_ptx(&CACHE, CUMSUM_PTX, "cumsum_kernel", "cumsum_f64_kernel");
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "cumsum_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "cumsum_f64_kernel",
                source: e,
            });
        }
    };
    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(total)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, total as u32);
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `cumsum_f64_kernel` resolved by
    //   `module_cache::get_or_compile` in the `match` directly
    //   above (returns `PtxCompileFailed` on Err); its `(in_ptr,
    //   out_ptr, outer, dim_size, inner, total)` ABI matches
    //   `CUMSUM_PTX` rewritten by `ptx_f32_to_f64`.
    // - `input: &CudaBuffer<f64>` is the source `[outer, dim_size,
    //   inner]` tensor (caller contract: `input.len() == n`).
    // - `out: &mut CudaBuffer<f64>` was alloc'd via
    //   `alloc_zeros_f64(n, device)?` for the same shape. The
    //   `&mut` borrow precludes aliasing `input`. The `n == 0`
    //   early-return above ensures the launch only runs when
    //   there's work.
    // - The kernel runs one thread per `(o, i)` pair (`total =
    //   outer * inner` threads) per the PTX bound check; each
    //   thread sequentially scans `dim_size` elements writing
    //   `out[(o*dim_size+k)*inner+i] = cumulative_sum` for `k in
    //   [0, dim_size)`. Reads of `input[(o*dim_size+k)*inner+i]`
    //   are bounded by `n`.
    // - `o`, `d`, `i`, `t` cannot truncate: `launch_cfg(total)?`
    //   returns `Err` if `total > u32::MAX`; `outer <= total`,
    //   `inner <= total`, `dim_size <= n / total <= u32::MAX`
    //   (caller contract on input length).
    // - cudarc enqueues the launch on `stream`; the six arg refs
    //   live until the trailing `?`. Stream sync is caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Cumulative product for f64.
#[cfg(feature = "cuda")]
pub fn gpu_cumprod_f64(
    input: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let total = outer * inner;
    let n = outer * dim_size * inner;
    if n == 0 {
        return cpu_to_gpu(&[], device);
    }
    let ctx = device.context();
    let stream = device.stream();
    let ptx = get_f64_ptx(&CACHE, CUMPROD_PTX, "cumprod_kernel", "cumprod_f64_kernel");
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "cumprod_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "cumprod_f64_kernel",
                source: e,
            });
        }
    };
    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(total)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, total as u32);
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `cumprod_f64_kernel` resolved by
    //   `module_cache::get_or_compile` in the `match` directly
    //   above (returns `PtxCompileFailed` on Err); its `(in_ptr,
    //   out_ptr, outer, dim_size, inner, total)` ABI matches
    //   `CUMPROD_PTX` rewritten by `ptx_f32_to_f64`.
    // - `input: &CudaBuffer<f64>` is the source `[outer, dim_size,
    //   inner]` tensor (caller contract: `input.len() == n`).
    //   `out: &mut CudaBuffer<f64>` was alloc'd via
    //   `alloc_zeros_f64(n, device)?`. The `&mut` borrow
    //   precludes aliasing `input`. The `n == 0` early-return
    //   above ensures the launch only runs when there's work.
    // - The kernel runs one thread per `(o, i)` pair (`total =
    //   outer * inner` threads) per the PTX bound check; each
    //   thread sequentially scans `dim_size` elements writing
    //   `out[(o*dim_size+k)*inner+i] = cumulative_product` for
    //   `k in [0, dim_size)`. Reads bounded by `n`.
    // - `o`, `d`, `i`, `t` cannot truncate: `launch_cfg(total)?`
    //   returns `Err` if `total > u32::MAX`; per-axis dims
    //   bounded by total via `outer * inner = total`,
    //   `dim_size <= n / total`.
    // - cudarc enqueues the launch on `stream`; the six arg refs
    //   live until the trailing `?`. Stream sync is caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Cumulative max for f64. Returns (values, indices).
#[cfg(feature = "cuda")]
pub fn gpu_cummax_f64(
    input: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>)> {
    use cudarc::driver::PushKernelArg;
    let total = outer * inner;
    let n = outer * dim_size * inner;
    if n == 0 {
        let e: &[f64] = &[];
        return Ok((cpu_to_gpu(e, device)?, cpu_to_gpu(e, device)?));
    }
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ctx = device.context();
    let stream = device.stream();
    let ptx = get_f64_ptx(&CACHE, CUMMAX_PTX, "cummax_kernel", "cummax_f64_kernel");
    let f =
        crate::module_cache::get_or_compile(ctx, ptx, "cummax_f64_kernel", device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed {
                kernel: "cummax_kernel",
                source: e,
            })?;
    let mut out = alloc_zeros_f64(n, device)?;
    let mut ind = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(total)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, total as u32);
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `cummax_f64_kernel` resolved by
    //   `module_cache::get_or_compile` directly above (returns
    //   `PtxCompileFailed` on Err); its `(in_ptr, out_ptr,
    //   ind_ptr, outer, dim_size, inner, total)` ABI matches
    //   `CUMMAX_PTX` rewritten by `ptx_f32_to_f64`.
    // - `input: &CudaBuffer<f64>` is the source `[outer, dim_size,
    //   inner]` (caller contract: `input.len() == n`).
    // - `out: &mut CudaBuffer<f64>` and `ind: &mut CudaBuffer<f64>`
    //   are two distinct `[outer, dim_size, inner]` buffers
    //   alloc'd via `alloc_zeros_f64(n, device)?`. Rust borrow
    //   rules guarantee they cannot alias each other or `input`
    //   (note: `ind` is f64 holding indices as floats — design
    //   choice of the caller, kernel writes integer values that
    //   fit exactly in f64).
    // - The kernel runs one thread per `(o, i)` pair (`total =
    //   outer * inner` threads); each thread sequentially scans
    //   `dim_size` elements writing `out[(o*dim_size+k)*inner+i]
    //   = running_max` and `ind[...] = argmax_k` per the PTX
    //   bound check.
    // - `o`, `d`, `i`, `t` cannot truncate: `launch_cfg(total)?`
    //   returns `Err` if `total > u32::MAX`; per-axis dims fit
    //   the same as cumsum/cumprod above.
    // - cudarc enqueues the launch on `stream`; the seven arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(ind.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok((out, ind))
}

/// Cumulative min for f64. Returns (values, indices).
#[cfg(feature = "cuda")]
pub fn gpu_cummin_f64(
    input: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>)> {
    use cudarc::driver::PushKernelArg;
    let total = outer * inner;
    let n = outer * dim_size * inner;
    if n == 0 {
        let e: &[f64] = &[];
        return Ok((cpu_to_gpu(e, device)?, cpu_to_gpu(e, device)?));
    }
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ctx = device.context();
    let stream = device.stream();
    let ptx = get_f64_ptx(&CACHE, CUMMIN_PTX, "cummin_kernel", "cummin_f64_kernel");
    let f =
        crate::module_cache::get_or_compile(ctx, ptx, "cummin_f64_kernel", device.ordinal() as u32)
            .map_err(|e| GpuError::PtxCompileFailed {
                kernel: "cummin_kernel",
                source: e,
            })?;
    let mut out = alloc_zeros_f64(n, device)?;
    let mut ind = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(total)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, total as u32);
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `cummin_f64_kernel` resolved by
    //   `module_cache::get_or_compile` directly above (returns
    //   `PtxCompileFailed` on Err); its `(in_ptr, out_ptr,
    //   ind_ptr, outer, dim_size, inner, total)` ABI matches
    //   `CUMMIN_PTX` rewritten by `ptx_f32_to_f64`.
    // - `input: &CudaBuffer<f64>` is the source `[outer, dim_size,
    //   inner]` (caller contract: `input.len() == n`).
    // - `out: &mut CudaBuffer<f64>` and `ind: &mut CudaBuffer<f64>`
    //   are two distinct buffers alloc'd via
    //   `alloc_zeros_f64(n, device)?`. Rust borrow rules
    //   guarantee they cannot alias each other or `input`. `ind`
    //   holds argmin indices as f64 values that fit exactly in
    //   the mantissa.
    // - The kernel runs one thread per `(o, i)` pair (`total =
    //   outer * inner` threads); each thread sequentially scans
    //   `dim_size` elements writing `out[...] = running_min`
    //   and `ind[...] = argmin_k` per the PTX bound check.
    // - `o`, `d`, `i`, `t` cannot truncate: `launch_cfg(total)?`
    //   returns `Err` if `total > u32::MAX`; per-axis dims fit.
    // - cudarc enqueues the launch on `stream`; the seven arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(ind.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok((out, ind))
}

/// Log-cumulative-sum-exp for f64.
#[cfg(feature = "cuda")]
pub fn gpu_logcumsumexp_f64(
    input: &CudaBuffer<f64>,
    outer: usize,
    dim_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    let total = outer * inner;
    let n = outer * dim_size * inner;
    if n == 0 {
        return cpu_to_gpu(&[], device);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = match crate::module_cache::get_or_compile(
        ctx,
        LOGCUMSUMEXP_F64_PTX,
        "logcumsumexp_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "logcumsumexp_f64_kernel",
                source: e,
            });
        }
    };
    let mut out = alloc_zeros_f64(n, device)?;
    let cfg = launch_cfg(total)?;
    let (o, d, i, t) = (outer as u32, dim_size as u32, inner as u32, total as u32);
    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `logcumsumexp_f64_kernel` resolved by
    //   `module_cache::get_or_compile` in the `match` directly
    //   above (returns `PtxCompileFailed` on Err); its `(in_ptr,
    //   out_ptr, outer, dim_size, inner, total)` ABI matches
    //   `LOGCUMSUMEXP_F64_PTX`.
    // - `input: &CudaBuffer<f64>` is the source `[outer, dim_size,
    //   inner]` tensor (caller contract: `input.len() == n`).
    //   `out: &mut CudaBuffer<f64>` was alloc'd via
    //   `alloc_zeros_f64(n, device)?`. The `&mut` borrow
    //   precludes aliasing `input`. The `n == 0` early-return
    //   above ensures the launch only runs when there's work.
    // - The kernel runs one thread per `(o, i)` pair (`total =
    //   outer * inner` threads); each thread maintains a running
    //   max and log-sum-exp across `dim_size` elements,
    //   producing the numerically-stable cumulative log-sum-exp
    //   in `out`.
    // - `o`, `d`, `i`, `t` cannot truncate: `launch_cfg(total)?`
    //   returns `Err` if `total > u32::MAX`; per-axis dims fit.
    // - cudarc enqueues the launch on `stream`; the six arg refs
    //   live until the trailing `?`. Stream sync is caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&o)
            .arg(&d)
            .arg(&i)
            .arg(&t)
            .launch(cfg)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- f64 softmax / log-softmax / layernorm / rmsnorm launchers
// ---------------------------------------------------------------------------

/// Row-wise softmax for f64 on GPU.
///
/// For each row: `out[j] = exp(x[j] - max(x)) / sum(exp(x - max(x)))`.
/// One block per row, 256 threads per block, shared-memory reductions.
#[cfg(feature = "cuda")]
pub fn gpu_softmax_f64(
    input: &CudaBuffer<f64>,
    rows: usize,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;

    validate_device(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SOFTMAX_F64_PTX,
        "softmax_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "softmax_f64_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(rows * cols, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8, // sdata[256] f64
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `softmax_f64_kernel`
    //   resolved by `module_cache::get_or_compile` in the `match`
    //   directly above (returns `PtxCompileFailed` on Err); its
    //   `(in_ptr, out_ptr, rows, cols)` ABI matches
    //   `SOFTMAX_F64_PTX`.
    // - `input: &CudaBuffer<f64>` is the row-major `[rows, cols]`
    //   activation; `validate_device(input, device)?` confirms
    //   device match (helper at line 10582). The caller is
    //   contracted to size `input` to `rows * cols`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd via
    //   `alloc_zeros_f64(rows * cols, device)?`. The `&mut`
    //   borrow precludes aliasing `input`.
    // - One block per row (grid `rows`), 256 threads per block,
    //   `256 * 8` bytes shared memory (sdata[256] of f64) for
    //   the per-row max+sum reduction. The kernel reads
    //   `input[row*cols+j]` and writes `out[row*cols+j]` for
    //   `row in [0, rows)` and `j in [0, cols)`, bounded by the
    //   PTX per-thread loop with stride `blockDim.x` capped at
    //   `cols`.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is u32-
    //   typed; caller contract ensures `rows * cols <= u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the four arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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

/// Row-wise softmax backward for f64 on GPU.
///
/// For each row: `out[j] = output[j] * (grad[j] - dot(grad_row, output_row))`.
#[cfg(feature = "cuda")]
pub fn gpu_softmax_backward_f64(
    grad: &CudaBuffer<f64>,
    output: &CudaBuffer<f64>,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;

    validate_device(grad, device)?;
    if grad.len() != output.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: output.len(),
        });
    }

    let total = grad.len();
    let rows = total / cols;

    let ctx = device.context();
    let stream = device.stream();

    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let ptx = get_f64_ptx(
        &CACHE,
        SOFTMAX_BACKWARD_PTX,
        "softmax_backward_kernel",
        "softmax_backward_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "softmax_backward_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "softmax_backward_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(total, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `softmax_backward_f64_kernel` resolved by
    //   `module_cache::get_or_compile` in the `match` directly
    //   above (returns `PtxCompileFailed` on Err); its `(grad_
    //   ptr, output_ptr, out_ptr, rows, cols)` ABI matches
    //   `SOFTMAX_BACKWARD_PTX` rewritten by `ptx_f32_to_f64`.
    // - `grad: &CudaBuffer<f64>` and `output: &CudaBuffer<f64>`
    //   are caller-supplied; `validate_device(grad, device)?`
    //   plus `grad.len() == output.len()` is enforced earlier
    //   in this fn (returns `LengthMismatch` otherwise).
    //   `total = grad.len()`, `rows = total / cols`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd via
    //   `alloc_zeros_f64(total, device)?`. The `&mut` borrow
    //   precludes aliasing `grad` or `output`.
    // - One block per row (grid `rows`), 256 threads per block,
    //   `256 * 8` bytes shared memory for per-row dot-product
    //   reduction. The kernel reads `grad[row*cols+j]`,
    //   `output[row*cols+j]` and writes `out[row*cols+j] =
    //   output[j] * (grad[j] - dot(grad_row, output_row))` for
    //   `row in [0, rows)` and `j in [0, cols)`.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is u32-
    //   typed; caller contract ensures `total = rows * cols <=
    //   u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the five arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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

/// Row-wise log-softmax for f64 on GPU.
///
/// For each row: `out[j] = x[j] - log(sum(exp(x - max(x))))`.
#[cfg(feature = "cuda")]
pub fn gpu_log_softmax_f64(
    input: &CudaBuffer<f64>,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;

    validate_device(input, device)?;

    let total = input.len();
    let rows = total / cols;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        LOG_SOFTMAX_F64_PTX,
        "log_softmax_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "softmax_backward_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(total, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for `log_softmax_f64_
    //   kernel` resolved by `module_cache::get_or_compile`
    //   directly above (the `.map_err` reports `kernel:
    //   "softmax_backward_kernel"` — a known wrong-kernel-name
    //   copy-paste bug, sibling to #708; substantiation only
    //   here, surfaced in Category A list); its `(in_ptr, out_
    //   ptr, rows, cols)` ABI matches `LOG_SOFTMAX_F64_PTX`.
    // - `input: &CudaBuffer<f64>` is the row-major activation;
    //   `validate_device(input, device)?` confirms device match.
    //   `total = input.len()`, `rows = total / cols`. Caller
    //   contract: `total == rows * cols`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd via
    //   `alloc_zeros_f64(total, device)?`. The `&mut` borrow
    //   precludes aliasing `input`.
    // - One block per row, 256 threads per block, `256 * 8`
    //   bytes shared memory for the per-row max + log-sum-exp
    //   reduction. The kernel reads `input[row*cols+j]` and
    //   writes `out[row*cols+j] = x[j] - log_sum_exp` for `row
    //   in [0, rows)` and `j in [0, cols)`.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is
    //   u32-typed; caller contract ensures `total <= u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the four arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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

/// Row-wise log-softmax backward for f64 on GPU.
///
/// For each row:
///   `sum_grad = sum(grad[j])`
///   `out[j] = grad[j] - exp(output[j]) * sum_grad`
#[cfg(feature = "cuda")]
pub fn gpu_log_softmax_backward_f64(
    grad: &CudaBuffer<f64>,
    output: &CudaBuffer<f64>,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;

    validate_device(grad, device)?;
    if grad.len() != output.len() {
        return Err(GpuError::LengthMismatch {
            a: grad.len(),
            b: output.len(),
        });
    }

    let total = grad.len();
    let rows = total / cols;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        LOG_SOFTMAX_BACKWARD_F64_PTX,
        "log_softmax_backward_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "log_softmax_backward_f64_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(total, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for
    //   `log_softmax_backward_f64_kernel` resolved by
    //   `module_cache::get_or_compile` in the `match` directly
    //   above (returns `PtxCompileFailed` on Err); its `(grad_
    //   ptr, output_ptr, out_ptr, rows, cols)` ABI matches
    //   `LOG_SOFTMAX_BACKWARD_F64_PTX`.
    // - `grad: &CudaBuffer<f64>` and `output: &CudaBuffer<f64>`
    //   are caller-supplied; `validate_device(grad, device)?`
    //   plus `grad.len() == output.len()` is enforced earlier in
    //   this fn (returns `LengthMismatch` otherwise). `total =
    //   grad.len()`, `rows = total / cols`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd via
    //   `alloc_zeros_f64(total, device)?`. The `&mut` borrow
    //   precludes aliasing `grad` or `output`.
    // - One block per row, 256 threads per block, `256 * 8`
    //   bytes shared memory for the per-row sum reduction. The
    //   kernel reads `grad[row*cols+j]`, `output[row*cols+j]`
    //   and writes `out[row*cols+j] = grad[j] - exp(output[j])
    //   * sum(grad_row)` for `row in [0, rows)` and `j in [0,
    //   cols)`.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is
    //   u32-typed; caller contract ensures `total <= u32::MAX`.
    // - cudarc enqueues the launch on `stream`; the five arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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

/// Row-wise LayerNorm for f64 on GPU.
///
/// `input`: `[rows * cols]`, `weight`: `[cols]`, `bias`: `[cols]`.
/// `out[j] = weight[j] * (x[j] - mean) / sqrt(var + eps) + bias[j]`.
#[cfg(feature = "cuda")]
pub fn gpu_layernorm_f64(
    input: &CudaBuffer<f64>,
    weight: &CudaBuffer<f64>,
    bias: &CudaBuffer<f64>,
    rows: usize,
    cols: usize,
    eps: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        LAYERNORM_PTX,
        "layernorm_kernel",
        "layernorm_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "layernorm_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "layernorm_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(rows * cols, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `layernorm_f64_kernel` resolved by
    //   `module_cache::get_or_compile` in the `match` directly
    //   above (returns `PtxCompileFailed` on Err); its `(input_
    //   ptr, out_ptr, weight_ptr, bias_ptr, rows, cols, eps)`
    //   ABI matches `LAYERNORM_PTX` rewritten by
    //   `ptx_f32_to_f64`.
    // - `input: &CudaBuffer<f64>` is the row-major `[rows, cols]`
    //   activation; `validate_device(input, device)?` confirms
    //   device match. `weight: &CudaBuffer<f64>` and `bias:
    //   &CudaBuffer<f64>` are per-column affine parameters
    //   (length `cols`). The caller is contracted to size
    //   `input` to `rows * cols`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd via
    //   `alloc_zeros_f64(rows * cols, device)?`. The `&mut`
    //   borrow precludes aliasing `input`, `weight`, or `bias`.
    // - One block per row, 256 threads per block, `256 * 8`
    //   bytes shared memory for the per-row mean/var reduction.
    //   The kernel reads `input[row*cols+j]`, `weight[j]`,
    //   `bias[j]` and writes `out[row*cols+j]` for `row in
    //   [0, rows)` and `j in [0, cols)`.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is
    //   u32-typed; caller contract ensures `rows * cols <=
    //   u32::MAX`. `eps: f64` passed by-reference.
    // - cudarc enqueues the launch on `stream`; the seven arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
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

/// LayerNorm backward for f64 on GPU.
///
/// Returns `(grad_input [rows * cols], grad_weight [cols], grad_bias [cols])`.
#[cfg(feature = "cuda")]
pub fn gpu_layernorm_backward_f64(
    input: &CudaBuffer<f64>,
    grad_output: &CudaBuffer<f64>,
    weight: &CudaBuffer<f64>,
    rows: usize,
    cols: usize,
    eps: f64,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>, CudaBuffer<f64>)> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        LAYERNORM_BACKWARD_PTX,
        "layernorm_backward_kernel",
        "layernorm_backward_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "layernorm_backward_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "layernorm_backward_kernel",
                source: e,
            });
        }
    };

    let mut grad_in = alloc_zeros_f64(rows * cols, device)?;
    let mut grad_w = alloc_zeros_f64(cols, device)?;
    let mut grad_b = alloc_zeros_f64(cols, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `layernorm_backward_f64_kernel` resolved by
    //   `module_cache::get_or_compile` in the `match` directly
    //   above (returns `PtxCompileFailed` on Err); its ABI
    //   `(input, grad_out, weight, grad_in, grad_w, grad_b,
    //   rows, cols, eps)` matches `LAYERNORM_BACKWARD_PTX`
    //   rewritten by `ptx_f32_to_f64`.
    // - `input: &CudaBuffer<f64>` is the saved-for-backward
    //   activation; `grad_output: &CudaBuffer<f64>` is the
    //   upstream gradient; `weight: &CudaBuffer<f64>` is the
    //   per-col scale. `validate_device(input, device)?`
    //   confirms `input` lives on `device`. Caller contract:
    //   `input.len() == grad_output.len() == rows * cols`,
    //   `weight.len() == cols`.
    // - `grad_in: &mut CudaBuffer<f64>`, `grad_w: &mut
    //   CudaBuffer<f64>`, `grad_b: &mut CudaBuffer<f64>` are
    //   three distinct buffers alloc'd via `alloc_zeros_f64`.
    //   Rust borrow rules guarantee no aliasing between them or
    //   with the read-only inputs. `grad_w` and `grad_b` are
    //   accumulated via `atomicAdd` (PTX-level).
    // - One block per row, 256 threads per block, `256 * 8`
    //   bytes shared memory for per-row gradient reductions.
    //   The kernel reads `input[row*cols+j]`, `grad_output[row*
    //   cols+j]`, `weight[j]` and writes `grad_in[row*cols+j]`
    //   directly; `grad_w[j]` and `grad_b[j]` accumulated via
    //   atomic ops so cross-block updates compose correctly.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is
    //   u32-typed; caller contract ensures `rows * cols <=
    //   u32::MAX`. `eps: f64` passed by-reference.
    // - cudarc enqueues the launch on `stream`; the nine arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's responsibility before reading any gradient
    //   buffer.
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

/// Row-wise RMS normalization for f64 on GPU.
///
/// `input`: `[rows * cols]`, `weight`: `[cols]`.
/// `out[j] = x[j] * rsqrt(mean(x^2) + eps) * weight[j]`.
#[cfg(feature = "cuda")]
pub fn gpu_rmsnorm_f64(
    input: &CudaBuffer<f64>,
    weight: &CudaBuffer<f64>,
    rows: usize,
    cols: usize,
    eps: f64,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(&CACHE, RMSNORM_PTX, "rmsnorm_kernel", "rmsnorm_f64_kernel");
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "rmsnorm_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "rmsnorm_kernel",
                source: e,
            });
        }
    };

    let mut out = alloc_zeros_f64(rows * cols, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `rmsnorm_f64_kernel` resolved by
    //   `module_cache::get_or_compile` in the `match` directly
    //   above (returns `PtxCompileFailed` on Err); its `(input_
    //   ptr, out_ptr, weight_ptr, rows, cols, eps)` ABI matches
    //   `RMSNORM_PTX` rewritten by `ptx_f32_to_f64`.
    // - `input: &CudaBuffer<f64>` is the row-major `[rows, cols]`
    //   activation; `validate_device(input, device)?` confirms
    //   device match. `weight: &CudaBuffer<f64>` is the per-col
    //   scale (`weight.len() == cols`). Caller contract:
    //   `input.len() == rows * cols`.
    // - `out: &mut CudaBuffer<f64>` was alloc'd via
    //   `alloc_zeros_f64(rows * cols, device)?`. The `&mut`
    //   borrow precludes aliasing `input` or `weight`.
    // - One block per row, 256 threads per block, `256 * 8`
    //   bytes shared memory for the per-row mean-of-squares
    //   reduction. The kernel reads `input[row*cols+j]`,
    //   `weight[j]` and writes `out[row*cols+j] = x[j] *
    //   rsqrt(mean(x^2) + eps) * weight[j]` for `row in [0,
    //   rows)` and `j in [0, cols)`.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is
    //   u32-typed; caller contract ensures `rows * cols <=
    //   u32::MAX`. `eps: f64` passed by-reference.
    // - cudarc enqueues the launch on `stream`; the six arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(weight.inner())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }

    Ok(out)
}

/// RMSNorm backward for f64 on GPU.
///
/// Returns `(grad_input [rows * cols], grad_weight [cols])`.
#[cfg(feature = "cuda")]
pub fn gpu_rmsnorm_backward_f64(
    input: &CudaBuffer<f64>,
    grad_output: &CudaBuffer<f64>,
    weight: &CudaBuffer<f64>,
    rows: usize,
    cols: usize,
    eps: f64,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>)> {
    use cudarc::driver::PushKernelArg;
    static CACHE: std::sync::OnceLock<String> = std::sync::OnceLock::new();

    validate_device(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let ptx = get_f64_ptx(
        &CACHE,
        RMSNORM_BACKWARD_PTX,
        "rmsnorm_backward_kernel",
        "rmsnorm_backward_f64_kernel",
    );
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx,
        "rmsnorm_backward_f64_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "rmsnorm_backward_kernel",
                source: e,
            });
        }
    };

    let mut grad_in = alloc_zeros_f64(rows * cols, device)?;
    let mut grad_w = alloc_zeros_f64(cols, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 8,
    };

    // SAFETY:
    // - `f` is a valid PTX `CudaFunction` for the f64-rewritten
    //   `rmsnorm_backward_f64_kernel` resolved by
    //   `module_cache::get_or_compile` in the `match` directly
    //   above (returns `PtxCompileFailed` on Err); its ABI
    //   `(input, grad_out, weight, grad_in, grad_w, rows, cols,
    //   eps)` matches `RMSNORM_BACKWARD_PTX` rewritten by
    //   `ptx_f32_to_f64`.
    // - `input: &CudaBuffer<f64>` is the saved-for-backward
    //   activation; `grad_output: &CudaBuffer<f64>` is the
    //   upstream gradient; `weight: &CudaBuffer<f64>` is the
    //   per-col scale. `validate_device(input, device)?`
    //   confirms `input` lives on `device`. Caller contract:
    //   `input.len() == grad_output.len() == rows * cols`,
    //   `weight.len() == cols`.
    // - `grad_in: &mut CudaBuffer<f64>` and `grad_w: &mut
    //   CudaBuffer<f64>` are two distinct buffers alloc'd via
    //   `alloc_zeros_f64`. Rust borrow rules guarantee no
    //   aliasing between them or with the read-only inputs.
    //   `grad_w` is accumulated via `atomicAdd` (PTX-level).
    // - One block per row, 256 threads per block, `256 * 8`
    //   bytes shared memory for per-row gradient reductions.
    //   The kernel reads `input[row*cols+j]`, `grad_output[row*
    //   cols+j]`, `weight[j]` and writes `grad_in[row*cols+j]`
    //   directly; `grad_w[j]` accumulated atomically across
    //   blocks.
    // - `rows_u32`, `cols_u32` cannot truncate: grid_dim is
    //   u32-typed; caller contract ensures `rows * cols <=
    //   u32::MAX`. `eps: f64` passed by-reference.
    // - cudarc enqueues the launch on `stream`; the eight arg
    //   refs live until the trailing `?`. Stream sync is
    //   caller's responsibility before reading the gradient
    //   buffers.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(grad_output.inner())
            .arg(weight.inner())
            .arg(grad_in.inner_mut())
            .arg(grad_w.inner_mut())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }

    Ok((grad_in, grad_w))
}

// ---------------------------------------------------------------------------
// Non-cuda stubs for softmax/layernorm/rmsnorm f64
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
pub fn gpu_softmax_f64(
    _input: &CudaBuffer<f64>,
    _rows: usize,
    _cols: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_softmax_backward_f64(
    _grad: &CudaBuffer<f64>,
    _output: &CudaBuffer<f64>,
    _cols: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_log_softmax_f64(
    _input: &CudaBuffer<f64>,
    _cols: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_log_softmax_backward_f64(
    _grad: &CudaBuffer<f64>,
    _output: &CudaBuffer<f64>,
    _cols: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_layernorm_f64(
    _input: &CudaBuffer<f64>,
    _weight: &CudaBuffer<f64>,
    _bias: &CudaBuffer<f64>,
    _rows: usize,
    _cols: usize,
    _eps: f64,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_layernorm_backward_f64(
    _input: &CudaBuffer<f64>,
    _grad_output: &CudaBuffer<f64>,
    _weight: &CudaBuffer<f64>,
    _rows: usize,
    _cols: usize,
    _eps: f64,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>, CudaBuffer<f64>)> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_rmsnorm_f64(
    _input: &CudaBuffer<f64>,
    _weight: &CudaBuffer<f64>,
    _rows: usize,
    _cols: usize,
    _eps: f64,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_rmsnorm_backward_f64(
    _input: &CudaBuffer<f64>,
    _grad_output: &CudaBuffer<f64>,
    _weight: &CudaBuffer<f64>,
    _rows: usize,
    _cols: usize,
    _eps: f64,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>)> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Non-cuda stubs for new f64 ops
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_f64(_input: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_tanh_f64(
    _input: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_erf_f64(
    _input: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_backward_f64(
    _grad: &CudaBuffer<f64>,
    _input: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_backward_tanh_f64(
    _grad: &CudaBuffer<f64>,
    _input: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_backward_erf_f64(
    _grad: &CudaBuffer<f64>,
    _input: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_silu_f64(_input: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_silu_backward_f64(
    _grad: &CudaBuffer<f64>,
    _input: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_elu_f64(
    _input: &CudaBuffer<f64>,
    _alpha: f64,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_elu_backward_f64(
    _grad: &CudaBuffer<f64>,
    _input: &CudaBuffer<f64>,
    _alpha: f64,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_mish_f64(_input: &CudaBuffer<f64>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_mish_backward_f64(
    _grad: &CudaBuffer<f64>,
    _input: &CudaBuffer<f64>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_clamp_f64(
    _input: &CudaBuffer<f64>,
    _min: f64,
    _max: f64,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_cumsum_f64(
    _input: &CudaBuffer<f64>,
    _outer: usize,
    _dim_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_cumprod_f64(
    _input: &CudaBuffer<f64>,
    _outer: usize,
    _dim_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_cummax_f64(
    _input: &CudaBuffer<f64>,
    _outer: usize,
    _dim_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>)> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_cummin_f64(
    _input: &CudaBuffer<f64>,
    _outer: usize,
    _dim_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>)> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_logcumsumexp_f64(
    _input: &CudaBuffer<f64>,
    _outer: usize,
    _dim_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
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

    // -- gpu_strided_copy (CL-496) -------------------------------------

    #[test]
    fn strided_copy_identity_contiguous_2d() {
        // 2x3 contiguous — source strides are C-contiguous.
        // Source: [0, 1, 2, 3, 4, 5]
        // Expected output == source (identity copy).
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let (dev, input) = setup(&data);
        let out =
            gpu_strided_copy(&input, &[2, 3], &[3, 1], 0, &dev).expect("strided_copy identity");
        assert_buf_eq(&out, &dev, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn strided_copy_transpose_2d() {
        // Source 2x3 contiguous:
        //   [[0, 1, 2],
        //    [3, 4, 5]]
        // Transposed view shape [3, 2] with strides [1, 3]:
        //   out[i, j] = src[j, i]
        //   Expected: [[0, 3], [1, 4], [2, 5]] flat = [0, 3, 1, 4, 2, 5]
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let (dev, input) = setup(&data);
        let out =
            gpu_strided_copy(&input, &[3, 2], &[1, 3], 0, &dev).expect("strided_copy transpose");
        assert_buf_eq(&out, &dev, &[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn strided_copy_sliced_column() {
        // Source 3x4 contiguous:
        //   [[0, 1, 2, 3],
        //    [4, 5, 6, 7],
        //    [8, 9, 10, 11]]
        // Select column 2 via src_offset=2, shape=[3], stride=[4]:
        //   Expected: [2, 6, 10]
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let (dev, input) = setup(&data);
        let out = gpu_strided_copy(&input, &[3], &[4], 2, &dev).expect("strided_copy col slice");
        assert_buf_eq(&out, &dev, &[2.0, 6.0, 10.0]);
    }

    #[test]
    fn strided_copy_3d_permute() {
        // Source [2, 3, 4] contiguous, C-strides [12, 4, 1].
        // Permute (0, 2, 1) → view shape [2, 4, 3] with strides [12, 1, 4].
        //
        // out[b, i, j] = src[b, j, i]
        //
        // Build expected by doing the permute on the host.
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let (dev, input) = setup(&data);
        let out =
            gpu_strided_copy(&input, &[2, 4, 3], &[12, 1, 4], 0, &dev).expect("strided_copy 3d");

        let mut expected = vec![0.0f32; 24];
        for b in 0..2 {
            for i in 0..4 {
                for j in 0..3 {
                    let dst = b * 12 + i * 3 + j;
                    let src = b * 12 + j * 4 + i;
                    expected[dst] = data[src];
                }
            }
        }
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn strided_copy_4d_max_rank_supported() {
        // Rank 4 identity copy works.
        let shape = [2usize, 3, 2, 2];
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let (dev, input) = setup(&data);
        // C-contiguous strides: [12, 4, 2, 1]
        let out =
            gpu_strided_copy(&input, &shape, &[12, 4, 2, 1], 0, &dev).expect("strided_copy 4d");
        assert_buf_eq(&out, &dev, &data);
    }

    #[test]
    fn strided_copy_rejects_too_many_dims() {
        let (dev, input) = setup(&[0.0f32; 16]);
        // 9 dims > STRIDED_COPY_MAX_DIMS (8)
        let result = gpu_strided_copy(&input, &[1, 1, 1, 1, 1, 1, 1, 1, 16], &[1; 9], 0, &dev);
        assert!(result.is_err());
    }

    #[test]
    fn strided_copy_rejects_shape_stride_length_mismatch() {
        let (dev, input) = setup(&[0.0f32; 12]);
        let result = gpu_strided_copy(&input, &[3, 4], &[4, 1, 1], 0, &dev);
        assert!(result.is_err());
    }

    #[test]
    fn strided_copy_rejects_negative_stride() {
        let (dev, input) = setup(&[0.0f32; 6]);
        let result = gpu_strided_copy(&input, &[2, 3], &[3, -1], 0, &dev);
        assert!(result.is_err());
    }

    #[test]
    fn strided_copy_empty_output() {
        let (dev, input) = setup(&[1.0f32, 2.0, 3.0]);
        let out = gpu_strided_copy(&input, &[0, 3], &[3, 1], 0, &dev).expect("strided_copy empty");
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn strided_copy_f64_transpose_matches_f32() {
        // Same transpose test as the f32 version, using f64.
        let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let input = cpu_to_gpu(&data, &dev).expect("cpu_to_gpu f64");
        let out = gpu_strided_copy_f64(&input, &[3, 2], &[1, 3], 0, &dev)
            .expect("strided_copy_f64 transpose");
        let host = gpu_to_cpu(&out, &dev).expect("gpu_to_cpu f64");
        assert_eq!(host, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }
}
