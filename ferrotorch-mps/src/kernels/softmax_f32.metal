// softmax_f32.metal — last-dim softmax for ferrotorch MPS backend (#626).
//
// Input layout: [rows, cols] contiguous f32.
// Each threadgroup handles one row (one softmax vector).
// Grid: (rows,) threadgroups of `next_pow2(min(cols, 1024))` threads. The
// in-kernel tree reduction (`stride = tcount / 2; stride >>= 1`) requires
// a pow-2 threadgroup width; the dispatcher rounds the width up. Inactive
// threads (`tid >= cols`) keep the sentinel `local_max = -INFINITY` /
// `local_sum = 0.0f` because the strided init loop short-circuits for
// them — the reduction reads those sentinels but they are identity
// elements for max / sum so they don't affect the result. See #1101 and
// `ferrotorch_mps::backend::pow2_tg_width` for the dispatcher contract.
//
// Numerically stable: subtract row max before exp.
// Matches PyTorch's torch.softmax(x, dim=-1) on MPS device.

#include <metal_stdlib>
using namespace metal;

kernel void softmax_f32(
    device const float* inp [[ buffer(0) ]],
    device       float* out [[ buffer(1) ]],
    constant     uint&  rows [[ buffer(2) ]],
    constant     uint&  cols [[ buffer(3) ]],
    uint2 tgid   [[ threadgroup_position_in_grid ]],
    uint  tid    [[ thread_index_in_threadgroup ]],
    uint  tcount [[ threads_per_threadgroup ]]
) {
    uint row = tgid.x;
    if (row >= rows) return;

    device const float* row_in  = inp + row * cols;
    device       float* row_out = out + row * cols;

    // Step 1: find row max (for numerical stability)
    threadgroup float smax[1024];
    float local_max = -INFINITY;
    for (uint i = tid; i < cols; i += tcount) {
        local_max = max(local_max, row_in[i]);
    }
    smax[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce across threads
    for (uint stride = tcount / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smax[tid] = max(smax[tid], smax[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = smax[0];

    // Step 2: sum of exp(x - max)
    threadgroup float ssum[1024];
    float local_sum = 0.0f;
    for (uint i = tid; i < cols; i += tcount) {
        local_sum += exp(row_in[i] - row_max);
    }
    ssum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tcount / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = ssum[0];

    // Step 3: write normalised output
    for (uint i = tid; i < cols; i += tcount) {
        row_out[i] = exp(row_in[i] - row_max) / row_sum;
    }
}
