//! Permanent regression sentinel for #802: narrow on GPU returns wrong
//! values via the D2H readback path.
//!
//! Pre-fix, `Tensor::to(Device::Cpu)` calls the backend `gpu_to_cpu`
//! readback which copies the *entire* underlying GPU buffer, then
//! constructs a CPU tensor via `from_storage` — which **resets**
//! `storage_offset` to 0 and recomputes C-contiguous strides for the
//! view shape. That drops the original `storage_offset()` and
//! `strides()` from any stride-view (narrow, transpose, permute,
//! select, …), so subsequent `.data()` reads `slice[0..numel]` of the
//! full buffer instead of the actual view's elements.
//!
//! Concrete pre-fix failure for `narrow(0, 2, 3)` on `[1,2,3,4,5]`:
//! the view metadata is `shape=[3], offset=2, strides=[1]`, but after
//! `.cpu()` the new tensor has `shape=[3], offset=0, strides=[1]` over
//! the full 5-element buffer — `.data()` returns `[1.0, 2.0, 3.0]`
//! instead of the expected `[3.0, 4.0, 5.0]`.
//!
//! Post-fix, the GPU→CPU readback materializes any non-trivial view
//! on-device via `strided_copy_{f32,f64}` before D2H, so the resulting
//! CPU tensor is a freshly-contiguous copy of exactly the view.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation::from_vec;
use ferrotorch_core::Device;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// 1-D `narrow(0, 2, 3)` — the canonical case from the cascade issue.
/// Pre-fix returns `[1.0, 2.0, 3.0]`; post-fix returns `[3.0, 4.0, 5.0]`.
#[test]
fn narrow_1d_offset_readback_f32() {
    ensure_cuda_backend();
    let cpu = from_vec::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("construct cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    let view = gpu.narrow(0, 2, 3).expect("narrow");

    // The narrowed GPU view has offset=2, shape=[3], strides=[1].
    assert_eq!(view.shape(), &[3]);
    assert_eq!(view.storage_offset(), 2);
    assert!(view.is_contiguous());

    let host = view.cpu().expect("gpu->cpu");
    let host_slice = host.data().expect("host slice");
    assert_eq!(
        host_slice,
        &[3.0_f32, 4.0, 5.0],
        "narrow GPU→CPU readback dropped storage_offset (#802)"
    );
}

/// Same test for f64 to cover the f64 dispatch path.
#[test]
fn narrow_1d_offset_readback_f64() {
    ensure_cuda_backend();
    let cpu = from_vec::<f64>(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]).expect("construct cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    let view = gpu.narrow(0, 2, 3).expect("narrow");
    let host = view.cpu().expect("gpu->cpu");
    let host_slice = host.data().expect("host slice");
    assert_eq!(host_slice, &[3.0_f64, 4.0, 5.0]);
}

/// 2-D narrow on the leading dimension — exercises a view whose
/// underlying storage extends both before and after the view's
/// elements (offset != 0 AND storage_len > numel).
#[test]
fn narrow_2d_outer_dim_f32() {
    ensure_cuda_backend();
    // 4x3 row-major: rows 0..4, columns 0..3.
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        0.0, 1.0, 2.0,
        3.0, 4.0, 5.0,
        6.0, 7.0, 8.0,
        9.0, 10.0, 11.0,
    ];
    let cpu = from_vec::<f32>(data, &[4, 3]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    // narrow rows [1..3] => 2x3 view at offset=3, shape=[2,3], strides=[3,1].
    let view = gpu.narrow(0, 1, 2).expect("narrow rows");
    let host = view.cpu().expect("gpu->cpu");
    let host_slice = host.data().expect("host slice");
    assert_eq!(host_slice, &[3.0_f32, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

/// 2-D narrow on the inner dimension — produces a non-contiguous view.
/// Pre-fix this would (a) read the full buffer, (b) reset offset to 0
/// and use C-contiguous strides for the view shape, returning the
/// first `numel` elements — wrong both for the offset and the stride.
#[test]
fn narrow_2d_inner_dim_noncontiguous_f32() {
    ensure_cuda_backend();
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        0.0, 1.0, 2.0, 3.0,
        4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0,
    ];
    let cpu = from_vec::<f32>(data, &[3, 4]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    // narrow cols [1..3] => 3x2 view at offset=1, shape=[3,2], strides=[4,1].
    let view = gpu.narrow(1, 1, 2).expect("narrow cols");
    assert!(
        !view.is_contiguous(),
        "narrow on inner dim should produce a non-contiguous view"
    );
    let host = view.cpu().expect("gpu->cpu");
    // After `.cpu()` the materialized view must itself be contiguous.
    assert!(host.is_contiguous());
    let host_slice = host.data().expect("host slice");
    assert_eq!(host_slice, &[1.0_f32, 2.0, 5.0, 6.0, 9.0, 10.0]);
}

/// Transpose-view (non-contiguous, offset==0) — must also be
/// materialized correctly on D2H.
#[test]
fn transpose_view_readback_f32() {
    ensure_cuda_backend();
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ];
    let cpu = from_vec::<f32>(data, &[2, 3]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    // transpose to 3x2 view; strides become [1, 3], offset=0.
    let view = gpu.transpose(0, 1).expect("transpose");
    assert!(!view.is_contiguous());
    let host = view.cpu().expect("gpu->cpu");
    let host_slice = host.data().expect("host slice");
    assert_eq!(host_slice, &[1.0_f32, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

/// Permute (3-D) — covers a higher-rank stride view.
#[test]
fn permute_view_3d_readback_f32() {
    ensure_cuda_backend();
    // 2x2x3 row-major; values are flat indices.
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu = from_vec::<f32>(data.clone(), &[2, 2, 3]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    // permute (0,1,2) -> (2,0,1): result shape [3,2,2].
    let view = gpu.permute(&[2, 0, 1]).expect("permute");
    assert_eq!(view.shape(), &[3, 2, 2]);
    let host = view.cpu().expect("gpu->cpu");
    let host_data = host.data_vec().expect("data_vec");

    // Reference: do the same permute on CPU and compare.
    let cpu_view = cpu.permute(&[2, 0, 1]).expect("cpu permute");
    let cpu_ref = cpu_view.data_vec().expect("cpu data_vec");
    assert_eq!(host_data, cpu_ref);
}

/// Multi-dim narrow chained — narrow twice on different axes to
/// exercise non-trivial offset and strides simultaneously.
#[test]
fn narrow_chained_multi_dim_f32() {
    ensure_cuda_backend();
    // 4x4 with values 0..16.
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = from_vec::<f32>(data, &[4, 4]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    // narrow rows [1..3], then cols [1..3] => 2x2 view at offset=5, strides=[4,1].
    let view = gpu
        .narrow(0, 1, 2)
        .expect("narrow rows")
        .narrow(1, 1, 2)
        .expect("narrow cols");
    let host = view.cpu().expect("gpu->cpu");
    let host_data = host.data_vec().expect("data_vec");
    assert_eq!(host_data, vec![5.0_f32, 6.0, 9.0, 10.0]);
}

/// Contiguous, zero-offset, full-buffer GPU → CPU — the fast path.
/// Verifies the fix does not regress contiguous tensors.
#[test]
fn contiguous_full_buffer_fast_path_f32() {
    ensure_cuda_backend();
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = from_vec::<f32>(data.clone(), &[4, 4]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    assert!(gpu.is_contiguous());
    assert_eq!(gpu.storage_offset(), 0);
    let back = gpu.cpu().expect("gpu->cpu");
    let back_slice = back.data().expect("host slice");
    assert_eq!(back_slice, data.as_slice());
}
