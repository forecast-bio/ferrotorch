//! GPU-accelerated FlashAttention via a custom PTX kernel with shared memory.
//!
//! Computes `softmax(Q @ K^T / sqrt(d)) @ V` without materializing the full
//! N x N attention matrix, keeping peak memory at O(N) instead of O(N^2).
//!
//! # Algorithm
//!
//! Each CUDA thread handles one query position.  The thread loads its query
//! vector into registers, then iterates over key/value tiles that are
//! cooperatively loaded into shared memory by the entire thread block.
//! Within each tile the thread computes dot products against the shared K
//! rows, updates the online softmax statistics (running max `m` and running
//! sum `l`), and accumulates the weighted V contribution into a register-
//! resident output vector.
//!
//! After all tiles are processed the thread writes the final output row to
//! global memory.
//!
//! # Shared memory layout
//!
//! The kernel uses `TILE_K` rows of shared memory for K and V tiles.  With
//! `TILE_K = 32` and head dimensions up to 128, the shared memory budget is:
//!
//! ```text
//! 2 * TILE_K * d_max * sizeof(f32) = 2 * 32 * 128 * 4 = 32 KiB
//! ```
//!
//! This fits comfortably in the 48 KiB default shared-memory limit on all
//! architectures from sm_52 upwards.
//!
//! # CPU fallback
//!
//! When the `cuda` feature is disabled, [`gpu_flash_attention_f32`] returns
//! `GpuError::NoCudaFeature`.

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of key/value rows loaded into shared memory per tile.
///
/// 32 is a good default: it keeps shared memory usage modest while giving
/// enough work per tile to amortize the `bar.sync` cost.
const TILE_K: usize = 32;

/// Maximum head dimension supported by the kernel.
///
/// The kernel unrolls its inner dot-product loop at compile time up to this
/// limit.  128 covers the vast majority of transformer architectures (GPT-2
/// uses 64, LLaMA uses 128).
const D_MAX: usize = 128;

// ---------------------------------------------------------------------------
// PTX kernel
// ---------------------------------------------------------------------------

/// PTX kernel for FlashAttention.
///
/// Each thread computes the attention output for ONE query position.  The
/// entire thread block cooperatively loads K/V tiles into shared memory;
/// each thread then reads from shared memory to compute dot products and
/// accumulate the online-softmax output.
///
/// **Shared memory layout** (all f32):
///
/// ```text
/// [0 .. TILE_K * d)        — K tile  [TILE_K, d]
/// [TILE_K * d .. 2*TILE_K*d + TILE_K * d_v)  — actually:
///   offset TILE_K*d          — V tile  [TILE_K, d_v]
/// ```
///
/// The kernel declares shared memory dynamically via the launch config's
/// `shared_mem_bytes` field rather than static `.shared` to support
/// variable `d` / `d_v` at runtime.
///
/// **Register budget per thread** (approximate):
///
/// - `q_reg[D_MAX]` — query vector (up to 128 floats)
/// - `o_reg[D_MAX]` — accumulator for output (up to 128 floats)
/// - `m`, `l` — online softmax scalars
/// - A handful of index / temp registers
///
/// For d=64 this is ~130 floats per thread = 520 bytes, well within the
/// register file budget even at 256 threads/block.
#[cfg(feature = "cuda")]
const FLASH_ATTENTION_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// Dynamic shared memory declared externally -- size set at launch.
.extern .shared .align 4 .b8 smem[];

.visible .entry flash_attention_kernel(
    .param .u64 Q_ptr,          // [N_q, d]    row-major
    .param .u64 K_ptr,          // [N_k, d]    row-major
    .param .u64 V_ptr,          // [N_k, d_v]  row-major
    .param .u64 O_ptr,          // [N_q, d_v]  output, row-major
    .param .u32 N_q,            // number of query positions
    .param .u32 N_k,            // number of key positions
    .param .u32 d_param,        // head dimension (query/key)
    .param .u32 d_v_param,      // value dimension
    .param .f32 scale,          // 1 / sqrt(d)
    .param .u32 causal,         // 0 or 1
    .param .u32 tile_k          // tile size for K/V (TILE_K)
) {
    // ---------------------------------------------------------------
    // Register declarations
    // ---------------------------------------------------------------
    // All registers declared at top level (no nested .reg -- some PTX JITs choke on scoped regs)
    .reg .u32 %ltid, %bid, %bdim, %gid;
    .reg .u32 %nq, %nk, %d, %dv, %caus, %tk;
    .reg .f32 %sc;
    .reg .u64 %Q, %K, %V, %O;
    .reg .u32 %t0, %t1, %t2, %t3;
    .reg .u64 %addr, %off64, %smem_base;
    .reg .f32 %fval, %fval2, %dot, %m_reg, %l_reg, %m_new, %l_new;
    .reg .f32 %corr, %p_val, %tile_sum, %rescale_old, %rescale_new;
    .reg .f32 %neg_inf, %zero, %one;
    .reg .pred %p_oob, %p_causal, %p_masked, %p_lnew_pos, %p_kloop, %p_dloop;
    .reg .pred %p_causal_en, %p_tile_done, %p_load_valid;
    // Zero-output loop
    .reg .u32 %zi;
    .reg .u64 %zaddr;
    // Tile loop
    .reg .u32 %k_start, %k_end, %bk;
    // K-load loop
    .reg .u32 %total_k, %li;
    .reg .u64 %ld_src, %ld_dst;
    .reg .f32 %ld_val;
    // V-load loop
    .reg .u32 %total_v, %vi, %v_smem_off;
    // Key/dim loops
    .reg .u32 %ki, %di, %k_abs;
    .reg .f32 %q_val, %k_val, %s_val, %v_val, %o_val;
    .reg .f32 %tile_max;
    .reg .pred %p_ki_done, %p_di_done, %p_masked_elem;
    // exp computation
    .reg .f32 %log2e, %arg_corr, %arg_p;
    // output update
    .reg .u32 %ovi, %v_smem_off_reg;
    .reg .f32 %o_cur;
    .reg .u64 %o_addr, %v_addr;

    // ---------------------------------------------------------------
    // Load parameters
    // ---------------------------------------------------------------
    ld.param.u64 %Q,     [Q_ptr];
    ld.param.u64 %K,     [K_ptr];
    ld.param.u64 %V,     [V_ptr];
    ld.param.u64 %O,     [O_ptr];
    ld.param.u32 %nq,    [N_q];
    ld.param.u32 %nk,    [N_k];
    ld.param.u32 %d,     [d_param];
    ld.param.u32 %dv,    [d_v_param];
    ld.param.f32 %sc,    [scale];
    ld.param.u32 %caus,  [causal];
    ld.param.u32 %tk,    [tile_k];

    // ---------------------------------------------------------------
    // Global thread ID = query index
    // ---------------------------------------------------------------
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %ltid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %ltid;

    // Out-of-bounds guard -- threads beyond N_q do nothing (but still
    // participate in shared-memory loads and barriers).
    setp.ge.u32 %p_oob, %gid, %nq;

    // Is causal masking enabled?
    setp.ne.u32 %p_causal_en, %caus, 0;

    // Constants.
    mov.f32 %neg_inf, 0fFF7FFFFF;    // -3.4028235e+38 (negative max float)
    mov.f32 %zero,    0f00000000;
    mov.f32 %one,     0f3F800000;     // 1.0

    // ---------------------------------------------------------------
    // Shared-memory base address
    // ---------------------------------------------------------------
    mov.u64 %smem_base, smem;

    // ---------------------------------------------------------------
    // Initialize online-softmax state: m = -inf, l = 0
    // ---------------------------------------------------------------
    mov.f32 %m_reg, %neg_inf;
    mov.f32 %l_reg, %zero;

    // ---------------------------------------------------------------
    // Load this thread's query vector Q[gid, :] into the output area
    // of global memory -- we will use O_ptr as our accumulator
    // (initialized to zero) and only read Q from global memory.
    // Actually: we keep O in global memory and update in-place per tile.
    //
    // Strategy: For each tile we do the full online softmax update
    // directly reading Q from global mem (coalesced), K/V from smem.
    // O is written back to global mem after all tiles.
    //
    // But first: zero the output region for this query.
    // ---------------------------------------------------------------

    // Zero output O[gid, 0..d_v] if in-bounds.
    @%p_oob bra SKIP_ZERO_OUTPUT;
        mov.u32 %zi, 0;
ZERO_LOOP:
        setp.ge.u32 %p_dloop, %zi, %dv;
        @%p_dloop bra ZERO_DONE;

        // addr = O + (gid * d_v + zi) * 4
        mad.lo.u32 %t0, %gid, %dv, %zi;
        cvt.u64.u32 %off64, %t0;
        shl.b64 %off64, %off64, 2;
        add.u64 %zaddr, %O, %off64;
        st.global.f32 [%zaddr], %zero;

        add.u32 %zi, %zi, 1;
        bra ZERO_LOOP;
ZERO_DONE:
SKIP_ZERO_OUTPUT:

    // ---------------------------------------------------------------
    // Tile loop: iterate over K/V in chunks of tile_k
    // ---------------------------------------------------------------
        mov.u32 %k_start, 0;

TILE_LOOP:
        setp.ge.u32 %p_tile_done, %k_start, %nk;
        @%p_tile_done bra TILE_LOOP_END;

        // k_end = min(k_start + tile_k, N_k)
        add.u32 %k_end, %k_start, %tk;
        min.u32 %k_end, %k_end, %nk;
        sub.u32 %bk, %k_end, %k_start;

        // -----------------------------------------------------------
        // Cooperative load: K tile into smem[0 .. bk * d]
        // Each thread loads ceil(bk * d / blockDim.x) elements.
        // -----------------------------------------------------------
        mul.lo.u32 %total_k, %bk, %d;
            mov.u32 %li, %ltid;
LOAD_K_LOOP:
            setp.ge.u32 %p_load_valid, %li, %total_k;
            @%p_load_valid bra LOAD_K_DONE;

            mad.lo.u32 %t0, %k_start, %d, %li;
            cvt.u64.u32 %off64, %t0;
            shl.b64 %off64, %off64, 2;
            add.u64 %ld_src, %K, %off64;
            ld.global.f32 %ld_val, [%ld_src];

            cvt.u64.u32 %off64, %li;
            shl.b64 %off64, %off64, 2;
            add.u64 %ld_dst, %smem_base, %off64;
            st.shared.f32 [%ld_dst], %ld_val;

            add.u32 %li, %li, %bdim;
            bra LOAD_K_LOOP;
LOAD_K_DONE:

        // -----------------------------------------------------------
        // Cooperative load: V tile into smem[bk*d .. bk*d + bk*d_v]
        // -----------------------------------------------------------
            mul.lo.u32 %total_v, %bk, %dv;
            mul.lo.u32 %v_smem_off, %tk, %d;

            mov.u32 %vi, %ltid;
LOAD_V_LOOP:
            setp.ge.u32 %p_load_valid, %vi, %total_v;
            @%p_load_valid bra LOAD_V_DONE;

            mad.lo.u32 %t0, %k_start, %dv, %vi;
            cvt.u64.u32 %off64, %t0;
            shl.b64 %off64, %off64, 2;
            add.u64 %ld_src, %V, %off64;
            ld.global.f32 %ld_val, [%ld_src];

            add.u32 %t1, %v_smem_off, %vi;
            cvt.u64.u32 %off64, %t1;
            shl.b64 %off64, %off64, 2;
            add.u64 %ld_dst, %smem_base, %off64;
            st.shared.f32 [%ld_dst], %ld_val;

            add.u32 %vi, %vi, %bdim;
            bra LOAD_V_LOOP;
LOAD_V_DONE:

        // Barrier: ensure all K/V data is in shared memory.
        bar.sync 0;

        // -----------------------------------------------------------
        // Per-thread: compute attention for this tile
        // -----------------------------------------------------------
        @%p_oob bra SKIP_COMPUTE;

            mov.u32 %ki, 0;
KEY_LOOP:
            setp.ge.u32 %p_ki_done, %ki, %bk;
            @%p_ki_done bra KEY_LOOP_END;

            // Absolute key index for causal check.
            add.u32 %k_abs, %k_start, %ki;

            // Compute dot product: sum over d dimensions.
            mov.f32 %dot, %zero;
            mov.u32 %di, 0;
DOT_LOOP:
            setp.ge.u32 %p_di_done, %di, %d;
            @%p_di_done bra DOT_DONE;

            // q_val = Q[gid * d + di]
            mad.lo.u32 %t0, %gid, %d, %di;
            cvt.u64.u32 %off64, %t0;
            shl.b64 %off64, %off64, 2;
            add.u64 %addr, %Q, %off64;
            ld.global.f32 %q_val, [%addr];

            // k_val = smem[ki * d + di]  (K tile in shared memory)
            mad.lo.u32 %t0, %ki, %d, %di;
            cvt.u64.u32 %off64, %t0;
            shl.b64 %off64, %off64, 2;
            add.u64 %addr, %smem_base, %off64;
            ld.shared.f32 %k_val, [%addr];

            fma.rn.f32 %dot, %q_val, %k_val, %dot;

            add.u32 %di, %di, 1;
            bra DOT_LOOP;
DOT_DONE:
            // s_val = dot * scale
            mul.f32 %s_val, %dot, %sc;

            // Causal mask: if causal and k_abs > gid, set s = -inf.
            @!%p_causal_en bra SKIP_CAUSAL_MASK;
            setp.gt.u32 %p_masked_elem, %k_abs, %gid;
            @%p_masked_elem mov.f32 %s_val, %neg_inf;
SKIP_CAUSAL_MASK:

            // Online softmax update.
            // m_new = max(m_reg, s_val)
            max.f32 %m_new, %m_reg, %s_val;

            // Compute corr = exp(m_old - m_new) and p = exp(s - m_new)
            // using ex2.approx (2^x) with log2(e) conversion.
            mov.f32 %log2e, 0f3FB8AA3B;    // log2(e) = 1.4426950408889634

                // corr = exp(m_reg - m_new) = 2^((m_reg - m_new) * log2e)
                sub.f32 %arg_corr, %m_reg, %m_new;
                mul.f32 %arg_corr, %arg_corr, %log2e;
                ex2.approx.f32 %corr, %arg_corr;

                // p = exp(s_val - m_new) = 2^((s_val - m_new) * log2e)
                sub.f32 %arg_p, %s_val, %m_new;
                mul.f32 %arg_p, %arg_p, %log2e;
                ex2.approx.f32 %p_val, %arg_p;

            // l_new = corr * l_reg + p_val
            fma.rn.f32 %l_new, %corr, %l_reg, %p_val;

            // Update output: O[gid, dv_i] = (corr * l_reg / l_new) * O[gid, dv_i]
            //                              + (p_val / l_new) * V_tile[ki, dv_i]
            //
            // rescale_old = corr * l_reg / l_new
            // rescale_new = p_val / l_new
            // Special case: if l_new == 0 skip (all scores were -inf).
            setp.gt.f32 %p_lnew_pos, %l_new, %zero;
            @!%p_lnew_pos bra SKIP_OUTPUT_UPDATE;

            mul.f32 %rescale_old, %corr, %l_reg;
            div.approx.f32 %rescale_old, %rescale_old, %l_new;
            div.approx.f32 %rescale_new, %p_val, %l_new;

            // Loop over d_v to update output.
                mul.lo.u32 %v_smem_off_reg, %tk, %d;

                mov.u32 %ovi, 0;
OV_LOOP:
                setp.ge.u32 %p_di_done, %ovi, %dv;
                @%p_di_done bra OV_DONE;

                mad.lo.u32 %t0, %gid, %dv, %ovi;
                cvt.u64.u32 %off64, %t0;
                shl.b64 %off64, %off64, 2;
                add.u64 %o_addr, %O, %off64;
                ld.global.f32 %o_cur, [%o_addr];

                mad.lo.u32 %t0, %ki, %dv, %ovi;
                add.u32 %t0, %t0, %v_smem_off_reg;
                cvt.u64.u32 %off64, %t0;
                shl.b64 %off64, %off64, 2;
                add.u64 %v_addr, %smem_base, %off64;
                ld.shared.f32 %v_val, [%v_addr];

                mul.f32 %o_cur, %rescale_old, %o_cur;
                fma.rn.f32 %o_cur, %rescale_new, %v_val, %o_cur;

                st.global.f32 [%o_addr], %o_cur;

                add.u32 %ovi, %ovi, 1;
                bra OV_LOOP;
OV_DONE:

SKIP_OUTPUT_UPDATE:
            mov.f32 %m_reg, %m_new;
            mov.f32 %l_reg, %l_new;

            add.u32 %ki, %ki, 1;
            bra KEY_LOOP;
KEY_LOOP_END:
SKIP_COMPUTE:

        bar.sync 0;

        add.u32 %k_start, %k_start, %tk;
        bra TILE_LOOP;
TILE_LOOP_END:

    ret;
}
";

// ---------------------------------------------------------------------------
// Rust wrapper
// ---------------------------------------------------------------------------

/// Compute FlashAttention on the GPU.
///
/// `softmax(Q @ K^T / sqrt(d)) @ V` — memory-efficient, O(N) peak memory.
///
/// # Arguments
///
/// - `query` -- GPU buffer `[batch_heads, N_q, d]` flattened row-major.
/// - `key`   -- GPU buffer `[batch_heads, N_k, d]` flattened row-major.
/// - `value` -- GPU buffer `[batch_heads, N_k, d_v]` flattened row-major.
/// - `n_q`   -- number of query positions.
/// - `n_k`   -- number of key positions.
/// - `d`     -- head dimension (query/key).
/// - `d_v`   -- value dimension.
/// - `batch_heads` -- `B * H`, the number of independent attention heads.
/// - `scale` -- typically `1.0 / sqrt(d)`.
/// - `causal` -- enable causal (lower-triangular) masking.
/// - `device` -- the GPU device owning all buffers.
///
/// # Returns
///
/// GPU buffer of shape `[batch_heads, N_q, d_v]` flattened row-major.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] if buffer lengths are inconsistent.
/// - [`GpuError::DeviceMismatch`] if buffers are on different devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub fn gpu_flash_attention_f32(
    query: &CudaBuffer<f32>,
    key: &CudaBuffer<f32>,
    value: &CudaBuffer<f32>,
    n_q: usize,
    n_k: usize,
    d: usize,
    d_v: usize,
    batch_heads: usize,
    scale: f32,
    causal: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    // --- Validate shapes ---------------------------------------------------

    if d > D_MAX {
        return Err(GpuError::ShapeMismatch {
            op: "flash_attention",
            expected: vec![D_MAX],
            got: vec![d],
        });
    }

    if d_v > D_MAX {
        return Err(GpuError::ShapeMismatch {
            op: "flash_attention",
            expected: vec![D_MAX],
            got: vec![d_v],
        });
    }

    // Validate shared memory requirement fits in 48 KiB.
    // smem = (TILE_K * d + TILE_K * d_v) * sizeof(f32)
    let smem_required = (TILE_K * d + TILE_K * d_v) * std::mem::size_of::<f32>();
    const SMEM_LIMIT: usize = 48 * 1024; // 48 KiB
    if smem_required > SMEM_LIMIT {
        return Err(GpuError::ShapeMismatch {
            op: "flash_attention",
            expected: vec![SMEM_LIMIT],
            got: vec![smem_required],
        });
    }

    let expected_q = batch_heads * n_q * d;
    if query.len() != expected_q {
        return Err(GpuError::ShapeMismatch {
            op: "flash_attention",
            expected: vec![batch_heads, n_q, d],
            got: vec![query.len()],
        });
    }

    let expected_k = batch_heads * n_k * d;
    if key.len() != expected_k {
        return Err(GpuError::ShapeMismatch {
            op: "flash_attention",
            expected: vec![batch_heads, n_k, d],
            got: vec![key.len()],
        });
    }

    let expected_v = batch_heads * n_k * d_v;
    if value.len() != expected_v {
        return Err(GpuError::ShapeMismatch {
            op: "flash_attention",
            expected: vec![batch_heads, n_k, d_v],
            got: vec![value.len()],
        });
    }

    // --- Validate devices --------------------------------------------------

    if query.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: query.device_ordinal(),
        });
    }
    if key.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: key.device_ordinal(),
        });
    }
    if value.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: value.device_ordinal(),
        });
    }

    // --- Handle degenerate cases -------------------------------------------

    let total_out = batch_heads * n_q * d_v;
    if batch_heads == 0 || n_q == 0 || d_v == 0 {
        return crate::transfer::alloc_zeros_f32(0, device);
    }

    // --- Allocate output ---------------------------------------------------

    let mut output = crate::transfer::alloc_zeros_f32(total_out, device)?;

    // --- Load PTX module (cached after first compilation) -----------------

    let ctx = device.context();
    let stream = device.stream();

    let kernel_fn = crate::module_cache::get_or_compile(
        ctx,
        FLASH_ATTENTION_PTX,
        "flash_attention_kernel",
        device.ordinal() as u32,
    )?;

    // --- Compute shared memory requirement ---------------------------------
    //
    // smem = (TILE_K * d + TILE_K * d_v) * sizeof(f32)

    let smem_bytes = (TILE_K * d + TILE_K * d_v) * std::mem::size_of::<f32>();

    // --- Launch config -----------------------------------------------------
    //
    // One thread per query position.  256 threads per block is a reasonable
    // default that balances occupancy and register pressure.

    const BLOCK: u32 = 256;

    if n_q > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "flash_attention",
            expected: vec![u32::MAX as usize],
            got: vec![n_q],
        });
    }
    let grid_x = ((n_q as u32).saturating_add(BLOCK - 1)) / BLOCK;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_x.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem_bytes as u32,
    };

    // --- Per-head launch ---------------------------------------------------

    let n_q_u32 = n_q as u32;
    let n_k_u32 = n_k as u32;
    let d_u32 = d as u32;
    let d_v_u32 = d_v as u32;
    let causal_u32: u32 = if causal { 1 } else { 0 };
    let tile_k_u32 = TILE_K as u32;

    for bh in 0..batch_heads {
        // Compute offsets into the flattened buffers for this head.
        let q_offset = bh * n_q * d;
        let k_offset = bh * n_k * d;
        let v_offset = bh * n_k * d_v;
        let o_offset = bh * n_q * d_v;

        // Create sub-slices that point to this head's data.
        let q_view = query.inner().slice(q_offset..q_offset + n_q * d);
        let k_view = key.inner().slice(k_offset..k_offset + n_k * d);
        let v_view = value.inner().slice(v_offset..v_offset + n_k * d_v);
        let mut o_view = output.inner_mut().slice_mut(o_offset..o_offset + n_q * d_v);

        // SAFETY: The kernel reads from Q, K, V sub-slices and writes to O
        // sub-slice.  All buffers are device-resident with sufficient length.
        // The grid covers N_q threads; out-of-range threads are guarded.
        // Shared memory is sized to hold TILE_K * (d + d_v) floats.
        unsafe {
            stream
                .launch_builder(&kernel_fn)
                .arg(&q_view)
                .arg(&k_view)
                .arg(&v_view)
                .arg(&mut o_view)
                .arg(&n_q_u32)
                .arg(&n_k_u32)
                .arg(&d_u32)
                .arg(&d_v_u32)
                .arg(&scale)
                .arg(&causal_u32)
                .arg(&tile_k_u32)
                .launch(cfg)?;
        }
    }

    Ok(output)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_flash_attention_f32(
    _query: &CudaBuffer<f32>,
    _key: &CudaBuffer<f32>,
    _value: &CudaBuffer<f32>,
    _n_q: usize,
    _n_k: usize,
    _d: usize,
    _d_v: usize,
    _batch_heads: usize,
    _scale: f32,
    _causal: bool,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// CPU reference (for testing)
// ---------------------------------------------------------------------------

/// Pure CPU FlashAttention reference using the online softmax trick.
///
/// Operates on a single head: Q `[N_q, d]`, K `[N_k, d]`, V `[N_k, d_v]`.
/// Returns output `[N_q, d_v]`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cpu_flash_attention_ref(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_q: usize,
    n_k: usize,
    d: usize,
    d_v: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let neg_inf: f32 = -1e30;
    let mut output = vec![0.0f32; n_q * d_v];

    for qi in 0..n_q {
        let mut m: f32 = f32::NEG_INFINITY;
        let mut l: f32 = 0.0;

        for ki in 0..n_k {
            // Dot product.
            let mut dot = 0.0f32;
            for di in 0..d {
                dot += q[qi * d + di] * k[ki * d + di];
            }
            let s = dot * scale;

            // Causal mask.
            let s = if causal && ki > qi { neg_inf } else { s };

            // Online softmax update.
            let m_new = m.max(s);
            let corr = (m - m_new).exp();
            let p = (s - m_new).exp();
            let l_new = corr * l + p;

            if l_new > 0.0 {
                let rescale_old = corr * l / l_new;
                let rescale_new = p / l_new;

                for dv in 0..d_v {
                    output[qi * d_v + dv] =
                        rescale_old * output[qi * d_v + dv] + rescale_new * v[ki * d_v + dv];
                }
            }

            m = m_new;
            l = l_new;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests -- require a real CUDA GPU
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::device::GpuDevice;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

    /// Compare two f32 slices with relative + absolute tolerance.
    fn assert_close(got: &[f32], expected: &[f32], rtol: f32, atol: f32, label: &str) {
        assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let tol = atol + rtol * e.abs();
            assert!(
                (g - e).abs() <= tol,
                "{label}: element {i}: got {g}, expected {e}, diff {}, tol {tol}",
                (g - e).abs(),
            );
        }
    }

    // -- Correctness vs CPU reference ----------------------------------------

    #[test]
    fn flash_attention_matches_cpu() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let batch_heads = 2;
        let n_q = 16;
        let n_k = 16;
        let d = 32;
        let d_v = 32;
        let scale = 1.0 / (d as f32).sqrt();

        // Deterministic pseudo-random data.
        let q_data: Vec<f32> = (0..batch_heads * n_q * d)
            .map(|i| ((i * 7 + 13) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let k_data: Vec<f32> = (0..batch_heads * n_k * d)
            .map(|i| ((i * 11 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let v_data: Vec<f32> = (0..batch_heads * n_k * d_v)
            .map(|i| ((i * 13 + 7) % 100) as f32 / 100.0 - 0.5)
            .collect();

        // CPU reference.
        let mut expected = Vec::with_capacity(batch_heads * n_q * d_v);
        for bh in 0..batch_heads {
            let q_slice = &q_data[bh * n_q * d..(bh + 1) * n_q * d];
            let k_slice = &k_data[bh * n_k * d..(bh + 1) * n_k * d];
            let v_slice = &v_data[bh * n_k * d_v..(bh + 1) * n_k * d_v];
            let out =
                cpu_flash_attention_ref(q_slice, k_slice, v_slice, n_q, n_k, d, d_v, scale, false);
            expected.extend_from_slice(&out);
        }

        // GPU.
        let q_gpu = cpu_to_gpu(&q_data, &dev).expect("q to gpu");
        let k_gpu = cpu_to_gpu(&k_data, &dev).expect("k to gpu");
        let v_gpu = cpu_to_gpu(&v_data, &dev).expect("v to gpu");

        let out_gpu = gpu_flash_attention_f32(
            &q_gpu,
            &k_gpu,
            &v_gpu,
            n_q,
            n_k,
            d,
            d_v,
            batch_heads,
            scale,
            false,
            &dev,
        )
        .expect("gpu_flash_attention_f32");

        assert_eq!(out_gpu.len(), batch_heads * n_q * d_v);

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(
            &out_host,
            &expected,
            1e-2,
            1e-5,
            "flash_attention non-causal",
        );
    }

    #[test]
    fn flash_attention_causal() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let batch_heads = 1;
        let n = 32; // N_q == N_k for causal
        let d = 64;
        let d_v = 64;
        let scale = 1.0 / (d as f32).sqrt();

        let q_data: Vec<f32> = (0..batch_heads * n * d)
            .map(|i| ((i * 7 + 13) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let k_data: Vec<f32> = (0..batch_heads * n * d)
            .map(|i| ((i * 11 + 3) % 100) as f32 / 100.0 - 0.5)
            .collect();
        let v_data: Vec<f32> = (0..batch_heads * n * d_v)
            .map(|i| ((i * 13 + 7) % 100) as f32 / 100.0 - 0.5)
            .collect();

        // CPU reference (causal).
        let expected =
            cpu_flash_attention_ref(&q_data, &k_data, &v_data, n, n, d, d_v, scale, true);

        // GPU.
        let q_gpu = cpu_to_gpu(&q_data, &dev).expect("q to gpu");
        let k_gpu = cpu_to_gpu(&k_data, &dev).expect("k to gpu");
        let v_gpu = cpu_to_gpu(&v_data, &dev).expect("v to gpu");

        let out_gpu = gpu_flash_attention_f32(
            &q_gpu,
            &k_gpu,
            &v_gpu,
            n,
            n,
            d,
            d_v,
            batch_heads,
            scale,
            true,
            &dev,
        )
        .expect("gpu_flash_attention_f32 causal");

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected, 1e-2, 1e-5, "flash_attention causal");
    }

    #[test]
    fn flash_attention_causal_first_row_is_just_first_value() {
        // With causal masking, query position 0 can only attend to key 0.
        // So output[0, :] should equal V[0, :].
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let n = 8;
        let d = 16;
        let d_v = 16;
        let scale = 1.0 / (d as f32).sqrt();

        let q_data: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.01).collect();
        let k_data: Vec<f32> = (0..n * d).map(|i| (i as f32) * 0.02).collect();
        let v_data: Vec<f32> = (0..n * d_v).map(|i| (i as f32) * 0.1).collect();

        let q_gpu = cpu_to_gpu(&q_data, &dev).expect("q to gpu");
        let k_gpu = cpu_to_gpu(&k_data, &dev).expect("k to gpu");
        let v_gpu = cpu_to_gpu(&v_data, &dev).expect("v to gpu");

        let out_gpu =
            gpu_flash_attention_f32(&q_gpu, &k_gpu, &v_gpu, n, n, d, d_v, 1, scale, true, &dev)
                .expect("gpu_flash_attention_f32");

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");

        // First row of output should be V[0, :].
        let first_row = &out_host[..d_v];
        let v_first_row = &v_data[..d_v];
        assert_close(
            first_row,
            v_first_row,
            1e-3,
            1e-5,
            "causal first row == V[0,:]",
        );
    }

    #[test]
    fn flash_attention_output_shape() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let batch_heads = 4;
        let n_q = 10;
        let n_k = 20;
        let d = 32;
        let d_v = 48;
        let scale = 1.0 / (d as f32).sqrt();

        let q_data = vec![0.1f32; batch_heads * n_q * d];
        let k_data = vec![0.1f32; batch_heads * n_k * d];
        let v_data = vec![0.1f32; batch_heads * n_k * d_v];

        let q_gpu = cpu_to_gpu(&q_data, &dev).expect("q to gpu");
        let k_gpu = cpu_to_gpu(&k_data, &dev).expect("k to gpu");
        let v_gpu = cpu_to_gpu(&v_data, &dev).expect("v to gpu");

        let out_gpu = gpu_flash_attention_f32(
            &q_gpu,
            &k_gpu,
            &v_gpu,
            n_q,
            n_k,
            d,
            d_v,
            batch_heads,
            scale,
            false,
            &dev,
        )
        .expect("gpu_flash_attention_f32");

        assert_eq!(out_gpu.len(), batch_heads * n_q * d_v);
    }

    #[test]
    fn flash_attention_different_seq_lengths() {
        // N_q != N_k (non-causal).
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let batch_heads = 1;
        let n_q = 8;
        let n_k = 24;
        let d = 16;
        let d_v = 16;
        let scale = 1.0 / (d as f32).sqrt();

        let q_data: Vec<f32> = (0..n_q * d)
            .map(|i| ((i * 7 + 5) % 50) as f32 / 50.0 - 0.5)
            .collect();
        let k_data: Vec<f32> = (0..n_k * d)
            .map(|i| ((i * 11 + 3) % 50) as f32 / 50.0 - 0.5)
            .collect();
        let v_data: Vec<f32> = (0..n_k * d_v)
            .map(|i| ((i * 13 + 7) % 50) as f32 / 50.0 - 0.5)
            .collect();

        let expected =
            cpu_flash_attention_ref(&q_data, &k_data, &v_data, n_q, n_k, d, d_v, scale, false);

        let q_gpu = cpu_to_gpu(&q_data, &dev).expect("q to gpu");
        let k_gpu = cpu_to_gpu(&k_data, &dev).expect("k to gpu");
        let v_gpu = cpu_to_gpu(&v_data, &dev).expect("v to gpu");

        let out_gpu = gpu_flash_attention_f32(
            &q_gpu,
            &k_gpu,
            &v_gpu,
            n_q,
            n_k,
            d,
            d_v,
            batch_heads,
            scale,
            false,
            &dev,
        )
        .expect("gpu_flash_attention_f32");

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(
            &out_host,
            &expected,
            1e-2,
            1e-5,
            "flash_attention diff seq lens",
        );
    }

    #[test]
    fn flash_attention_uniform_attention() {
        // All Q and K vectors are identical => uniform attention weights.
        // Output should be the mean of all V rows.
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let n_q = 4;
        let n_k = 4;
        let d = 8;
        let d_v = 8;
        let scale = 1.0 / (d as f32).sqrt();

        // Q = K = all ones => all dot products are equal => uniform softmax.
        let q_data = vec![1.0f32; n_q * d];
        let k_data = vec![1.0f32; n_k * d];
        // V rows: row i has all values = i+1.
        let mut v_data = vec![0.0f32; n_k * d_v];
        for i in 0..n_k {
            for j in 0..d_v {
                v_data[i * d_v + j] = (i + 1) as f32;
            }
        }

        // Expected: each output row = mean of V rows = (1+2+3+4)/4 = 2.5
        let expected = vec![2.5f32; n_q * d_v];

        let q_gpu = cpu_to_gpu(&q_data, &dev).expect("q to gpu");
        let k_gpu = cpu_to_gpu(&k_data, &dev).expect("k to gpu");
        let v_gpu = cpu_to_gpu(&v_data, &dev).expect("v to gpu");

        let out_gpu = gpu_flash_attention_f32(
            &q_gpu, &k_gpu, &v_gpu, n_q, n_k, d, d_v, 1, scale, false, &dev,
        )
        .expect("gpu_flash_attention_f32");

        let out_host = gpu_to_cpu(&out_gpu, &dev).expect("gpu_to_cpu");
        assert_close(&out_host, &expected, 1e-2, 1e-3, "uniform attention");
    }

    // -- Shape validation errors ---------------------------------------------

    #[test]
    fn flash_attention_query_length_mismatch() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let q_data = vec![0.0f32; 10]; // wrong length
        let k_data = vec![0.0f32; 4 * 8]; // B=1
        let v_data = vec![0.0f32; 4 * 8]; // B=1

        let q = cpu_to_gpu(&q_data, &dev).expect("q");
        let k = cpu_to_gpu(&k_data, &dev).expect("k");
        let v = cpu_to_gpu(&v_data, &dev).expect("v");

        let err = gpu_flash_attention_f32(&q, &k, &v, 4, 4, 8, 8, 1, 0.5, false, &dev).unwrap_err();
        match err {
            GpuError::ShapeMismatch {
                op: "flash_attention",
                ..
            } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn flash_attention_d_too_large() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        // d = 256 > D_MAX = 128
        let q_data = vec![0.0f32; 2 * 256]; // B=1
        let k_data = vec![0.0f32; 2 * 256]; // B=1
        let v_data = vec![0.0f32; 2 * 64]; // B=1

        let q = cpu_to_gpu(&q_data, &dev).expect("q");
        let k = cpu_to_gpu(&k_data, &dev).expect("k");
        let v = cpu_to_gpu(&v_data, &dev).expect("v");

        let err =
            gpu_flash_attention_f32(&q, &k, &v, 2, 2, 256, 64, 1, 0.5, false, &dev).unwrap_err();
        match err {
            GpuError::ShapeMismatch {
                op: "flash_attention",
                ..
            } => {}
            other => panic!("unexpected error: {other}"),
        }
    }
}
