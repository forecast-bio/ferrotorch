// sum_axis_f32.metal — reduction along one axis for ferrotorch MPS backend (#626).
//
// Reduces input[outer, axis_len, inner] → output[outer, inner] by summing
// along the axis_len dimension.
//
// outer  = product of dims before axis
// inner  = product of dims after axis
//
// Each threadgroup handles one (outer, inner) output cell.
// Grid: (outer * inner,) threadgroups of `next_pow2(min(axis_len, 1024))`
// threads. The in-kernel tree reduction (`stride = tcount / 2; stride >>= 1`)
// requires a pow-2 threadgroup width; the dispatcher rounds the width up.
// Inactive threads (`tid >= axis_len`) keep the sentinel `local = 0.0f`
// because the strided init loop short-circuits for them — the reduction
// reads those sentinels but `0.0f` is the identity element for sum so it
// doesn't affect the result. See #1101 and
// `ferrotorch_mps::backend::pow2_tg_width` for the dispatcher contract.
//
// Matches PyTorch's torch.sum(x, dim=axis) on MPS device (f32).

#include <metal_stdlib>
using namespace metal;

kernel void sum_axis_f32(
    device const float* inp      [[ buffer(0) ]],
    device       float* out      [[ buffer(1) ]],
    constant     uint&  outer    [[ buffer(2) ]],
    constant     uint&  axis_len [[ buffer(3) ]],
    constant     uint&  inner    [[ buffer(4) ]],
    uint2 tgid   [[ threadgroup_position_in_grid ]],
    uint  tid    [[ thread_index_in_threadgroup ]],
    uint  tcount [[ threads_per_threadgroup ]]
) {
    // Decode flat threadgroup index into (o, i)
    uint flat = tgid.x;
    if (flat >= outer * inner) return;
    uint o = flat / inner;
    uint i = flat % inner;

    threadgroup float ssum[1024];
    float local = 0.0f;
    for (uint a = tid; a < axis_len; a += tcount) {
        local += inp[o * axis_len * inner + a * inner + i];
    }
    ssum[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tcount / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        out[o * inner + i] = ssum[0];
    }
}
