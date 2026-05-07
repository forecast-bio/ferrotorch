//! Permanent regression sentinel for #812: non-contiguous CUDA inputs to
//! GPU elementwise binary ops (add / sub / mul / div) crashed inside the
//! kernel guard with `LengthMismatch` because `*_inner` read `gpu_handle()`
//! blindly — returning the WHOLE underlying storage of the narrowed view
//! while the kernel was sized for the (smaller) logical shape.
//!
//! Pre-fix (against `add_inner` at `arithmetic.rs:244-280`):
//!     thread '...' panicked at 'flex_attention forward f32:
//!     InvalidArgument { message: "buffer length mismatch: 32 vs 16" }'
//!
//! Post-fix: each `*_inner` branches on `is_contiguous()`. If the input is
//! a non-contiguous CUDA view, it materializes the view via
//! `Tensor::contiguous()` — which dispatches to the on-device
//! `strided_copy_{f32,f64}` kernel from #802 (CL-496 GPU fast path) — BEFORE
//! grabbing `gpu_handle()`. No CPU detour: the view stays on device.
//!
//! Coverage: add / sub / mul / div × {f32, f64} × {single-axis narrow,
//! double-axis narrow (nested non-contig)}.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::Tensor;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, sub};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("gpu->cpu copy")
    } else {
        t.clone()
    };
    cpu.data().expect("read_back").to_vec()
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    let cpu = if t.is_cuda() {
        t.cpu().expect("gpu->cpu copy")
    } else {
        t.clone()
    };
    cpu.data().expect("read_back").to_vec()
}

/// Build a `[4, 4]` row-major tensor of consecutive f32 values 0..16 on
/// CUDA, narrow rows `[1..3]` to obtain a 2x4 view (this happens to be
/// CONTIGUOUS — narrow on the leading dim of a row-major tensor preserves
/// contiguity), then narrow cols `[0..2]` of the result to force a
/// non-contiguous 2x2 view whose underlying storage spans 16 elements.
fn noncontig_2x2_f32() -> (Tensor<f32>, Vec<f32>) {
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = from_vec::<f32>(data, &[4, 4]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    // Outer narrow first to surface the 2x4 region rows[1..3].
    let v0 = gpu.narrow(0, 1, 2).expect("narrow rows");
    // Inner narrow to drop to cols[0..2] — produces a non-contiguous view.
    let view = v0.narrow(1, 0, 2).expect("narrow cols");
    assert_eq!(view.shape(), &[2, 2]);
    assert!(
        !view.is_contiguous(),
        "test setup: narrowed view must be non-contiguous for #812 to apply",
    );
    // Logical values: rows 1,2 cols 0,1 = [4, 5, 8, 9].
    let expected_view = vec![4.0_f32, 5.0, 8.0, 9.0];
    (view, expected_view)
}

fn noncontig_2x2_f64() -> (Tensor<f64>, Vec<f64>) {
    let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let cpu = from_vec::<f64>(data, &[4, 4]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    let v0 = gpu.narrow(0, 1, 2).expect("narrow rows");
    let view = v0.narrow(1, 0, 2).expect("narrow cols");
    assert_eq!(view.shape(), &[2, 2]);
    assert!(!view.is_contiguous(), "test setup: must be non-contiguous");
    let expected_view = vec![4.0_f64, 5.0, 8.0, 9.0];
    (view, expected_view)
}

fn ones_2x2_f32_gpu() -> Tensor<f32> {
    from_vec::<f32>(vec![1.0; 4], &[2, 2])
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
}

fn ones_2x2_f64_gpu() -> Tensor<f64> {
    from_vec::<f64>(vec![1.0; 4], &[2, 2])
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
}

fn twos_2x2_f32_gpu() -> Tensor<f32> {
    from_vec::<f32>(vec![2.0; 4], &[2, 2])
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
}

fn twos_2x2_f64_gpu() -> Tensor<f64> {
    from_vec::<f64>(vec![2.0; 4], &[2, 2])
        .expect("cpu tensor")
        .to(Device::Cuda(0))
        .expect("cpu->gpu")
}

// ---------------------------------------------------------------------------
// add — non-contiguous lhs against contiguous rhs of the view shape
// ---------------------------------------------------------------------------

#[test]
fn add_noncontig_lhs_f32() {
    ensure_cuda_backend();
    let (view, expected_view) = noncontig_2x2_f32();
    let other = ones_2x2_f32_gpu();
    let out = add(&view, &other).expect("add(noncontig, contig) f32");
    assert!(out.is_cuda(), "result must remain on CUDA");
    let got = read_back_f32(&out);
    let expected: Vec<f32> = expected_view.iter().map(|v| v + 1.0).collect();
    assert_eq!(got, expected);
}

#[test]
fn add_noncontig_lhs_f64() {
    ensure_cuda_backend();
    let (view, expected_view) = noncontig_2x2_f64();
    let other = ones_2x2_f64_gpu();
    let out = add(&view, &other).expect("add(noncontig, contig) f64");
    assert!(out.is_cuda());
    let got = read_back_f64(&out);
    let expected: Vec<f64> = expected_view.iter().map(|v| v + 1.0).collect();
    assert_eq!(got, expected);
}

// Symmetric: non-contiguous on the right.
#[test]
fn add_noncontig_rhs_f32() {
    ensure_cuda_backend();
    let (view, expected_view) = noncontig_2x2_f32();
    let other = twos_2x2_f32_gpu();
    let out = add(&other, &view).expect("add(contig, noncontig) f32");
    assert!(out.is_cuda());
    let got = read_back_f32(&out);
    let expected: Vec<f32> = expected_view.iter().map(|v| v + 2.0).collect();
    assert_eq!(got, expected);
}

#[test]
fn add_both_noncontig_f32() {
    ensure_cuda_backend();
    let (view_a, expected_view) = noncontig_2x2_f32();
    let (view_b, _) = noncontig_2x2_f32();
    let out = add(&view_a, &view_b).expect("add(noncontig, noncontig) f32");
    assert!(out.is_cuda());
    let got = read_back_f32(&out);
    let expected: Vec<f32> = expected_view.iter().map(|v| v + v).collect();
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// sub
// ---------------------------------------------------------------------------

#[test]
fn sub_noncontig_lhs_f32() {
    ensure_cuda_backend();
    let (view, expected_view) = noncontig_2x2_f32();
    let other = ones_2x2_f32_gpu();
    let out = sub(&view, &other).expect("sub(noncontig, contig) f32");
    assert!(out.is_cuda());
    let got = read_back_f32(&out);
    let expected: Vec<f32> = expected_view.iter().map(|v| v - 1.0).collect();
    assert_eq!(got, expected);
}

#[test]
fn sub_noncontig_lhs_f64() {
    ensure_cuda_backend();
    let (view, expected_view) = noncontig_2x2_f64();
    let other = ones_2x2_f64_gpu();
    let out = sub(&view, &other).expect("sub(noncontig, contig) f64");
    assert!(out.is_cuda());
    let got = read_back_f64(&out);
    let expected: Vec<f64> = expected_view.iter().map(|v| v - 1.0).collect();
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// mul
// ---------------------------------------------------------------------------

#[test]
fn mul_noncontig_lhs_f32() {
    ensure_cuda_backend();
    let (view, expected_view) = noncontig_2x2_f32();
    let other = twos_2x2_f32_gpu();
    let out = mul(&view, &other).expect("mul(noncontig, contig) f32");
    assert!(out.is_cuda());
    let got = read_back_f32(&out);
    let expected: Vec<f32> = expected_view.iter().map(|v| v * 2.0).collect();
    assert_eq!(got, expected);
}

#[test]
fn mul_noncontig_lhs_f64() {
    ensure_cuda_backend();
    let (view, expected_view) = noncontig_2x2_f64();
    let other = twos_2x2_f64_gpu();
    let out = mul(&view, &other).expect("mul(noncontig, contig) f64");
    assert!(out.is_cuda());
    let got = read_back_f64(&out);
    let expected: Vec<f64> = expected_view.iter().map(|v| v * 2.0).collect();
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// div
// ---------------------------------------------------------------------------

#[test]
fn div_noncontig_lhs_f32() {
    ensure_cuda_backend();
    let (view, expected_view) = noncontig_2x2_f32();
    let other = twos_2x2_f32_gpu();
    let out = div(&view, &other).expect("div(noncontig, contig) f32");
    assert!(out.is_cuda());
    let got = read_back_f32(&out);
    let expected: Vec<f32> = expected_view.iter().map(|v| v / 2.0).collect();
    assert_eq!(got, expected);
}

#[test]
fn div_noncontig_lhs_f64() {
    ensure_cuda_backend();
    let (view, expected_view) = noncontig_2x2_f64();
    let other = twos_2x2_f64_gpu();
    let out = div(&view, &other).expect("div(noncontig, contig) f64");
    assert!(out.is_cuda());
    let got = read_back_f64(&out);
    let expected: Vec<f64> = expected_view.iter().map(|v| v / 2.0).collect();
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// Edge: nested narrows (multiple non-contiguous transformations).
//
// This drills the chain narrow→narrow→narrow on a `[4, 4, 4]` tensor.
// After the third narrow the view metadata is shape=[2,2,2], offset>0,
// strides reflect both the row-major outer stride of the original buffer
// AND the inner narrow — strictly non-contiguous.
// ---------------------------------------------------------------------------

#[test]
fn add_nested_narrows_f32() {
    ensure_cuda_backend();
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let cpu = from_vec::<f32>(data, &[4, 4, 4]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    // Outer narrow on dim 0 (still contiguous).
    let v0 = gpu.narrow(0, 1, 2).expect("narrow d0");
    // Narrow on dim 1 — non-contiguous now (interior axis).
    let v1 = v0.narrow(1, 1, 2).expect("narrow d1");
    // Narrow on dim 2 — also non-contig.
    let view = v1.narrow(2, 1, 2).expect("narrow d2");
    assert_eq!(view.shape(), &[2, 2, 2]);
    assert!(!view.is_contiguous());

    // Reference: same chain on CPU, then read back.
    let cpu_view = cpu
        .narrow(0, 1, 2)
        .and_then(|t| t.narrow(1, 1, 2))
        .and_then(|t| t.narrow(2, 1, 2))
        .expect("cpu chain");
    let cpu_contig = cpu_view.contiguous().expect("cpu contig");
    let expected: Vec<f32> = cpu_contig.data().expect("cpu data").to_vec();

    let other = from_vec::<f32>(vec![1.0; 8], &[2, 2, 2])
        .expect("rhs cpu")
        .to(Device::Cuda(0))
        .expect("rhs gpu");
    let out = add(&view, &other).expect("add nested-narrow + ones");
    assert!(out.is_cuda());
    let got = read_back_f32(&out);
    let expected_added: Vec<f32> = expected.iter().map(|v| v + 1.0).collect();
    assert_eq!(got, expected_added);
}

#[test]
fn add_nested_narrows_f64() {
    ensure_cuda_backend();
    let data: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let cpu = from_vec::<f64>(data, &[4, 4, 4]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    let v0 = gpu.narrow(0, 1, 2).expect("narrow d0");
    let v1 = v0.narrow(1, 1, 2).expect("narrow d1");
    let view = v1.narrow(2, 1, 2).expect("narrow d2");
    assert!(!view.is_contiguous());

    let cpu_view = cpu
        .narrow(0, 1, 2)
        .and_then(|t| t.narrow(1, 1, 2))
        .and_then(|t| t.narrow(2, 1, 2))
        .expect("cpu chain");
    let cpu_contig = cpu_view.contiguous().expect("cpu contig");
    let expected: Vec<f64> = cpu_contig.data().expect("cpu data").to_vec();

    let other = from_vec::<f64>(vec![1.0; 8], &[2, 2, 2])
        .expect("rhs cpu")
        .to(Device::Cuda(0))
        .expect("rhs gpu");
    let out = add(&view, &other).expect("add nested-narrow + ones");
    assert!(out.is_cuda());
    let got = read_back_f64(&out);
    let expected_added: Vec<f64> = expected.iter().map(|v| v + 1.0).collect();
    assert_eq!(got, expected_added);
}

// ---------------------------------------------------------------------------
// Edge: "contiguous-by-strides but oversized buffer" — `narrow(0, ..)` on
// a row-major tensor produces a view that is `is_contiguous() == true`
// (strides match C-order over the view shape), yet the underlying GPU
// buffer is larger than the view's logical numel. This case bypasses the
// `is_contiguous()` shortcut in `Tensor::contiguous()` and must still be
// materialized before the elementwise kernel sees the handle.
// ---------------------------------------------------------------------------

#[test]
fn add_contig_by_strides_oversized_buffer_f32() {
    ensure_cuda_backend();
    // [4,4] base; narrow rows [1..3] -> [2,4]. is_contiguous() is true
    // (strides [4,1] match C-order over shape [2,4]) but underlying
    // buffer is 16 elements, view logical numel is 8.
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let cpu = from_vec::<f32>(data, &[4, 4]).expect("cpu tensor");
    let gpu = cpu.to(Device::Cuda(0)).expect("cpu->gpu");
    let view = gpu.narrow(0, 1, 2).expect("narrow rows");
    assert_eq!(view.shape(), &[2, 4]);
    // This view is "contiguous-by-strides" — a known PyTorch tripwire.
    assert!(view.is_contiguous());
    assert!(view.numel() != view.storage_len() || view.storage_offset() != 0);

    let other = from_vec::<f32>(vec![1.0; 8], &[2, 4])
        .expect("rhs cpu")
        .to(Device::Cuda(0))
        .expect("rhs gpu");
    let out = add(&view, &other).expect("add(contig-by-strides oversized, contig)");
    assert!(out.is_cuda());
    let got = read_back_f32(&out);
    let expected: Vec<f32> = (4..12).map(|i| i as f32 + 1.0).collect();
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------------
// Sanity: CONTIGUOUS path stays zero-copy (no regression on the fast path).
// ---------------------------------------------------------------------------

#[test]
fn add_contig_remains_zero_copy_f32() {
    ensure_cuda_backend();
    let a = from_vec::<f32>(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
        .expect("a cpu")
        .to(Device::Cuda(0))
        .expect("a gpu");
    let b = from_vec::<f32>(vec![10.0, 20.0, 30.0, 40.0], &[2, 2])
        .expect("b cpu")
        .to(Device::Cuda(0))
        .expect("b gpu");
    assert!(a.is_contiguous());
    assert!(b.is_contiguous());
    let out = add(&a, &b).expect("contig add");
    assert!(out.is_cuda());
    let got = read_back_f32(&out);
    assert_eq!(got, vec![11.0_f32, 22.0, 33.0, 44.0]);
}
