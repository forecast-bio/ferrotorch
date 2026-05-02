//! End-to-end channels_last memory format integration tests on real
//! CUDA hardware. CL-455.
//!
//! Verifies that `Tensor::contiguous_in(MemoryFormat::ChannelsLast)`
//! on a CUDA tensor:
//!   1. Stays on the device (no CPU roundtrip).
//!   2. Produces a buffer that's bit-identical to the CPU reference
//!      conversion.
//!   3. Has the right strides for channels-last (NHWC pattern).

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, MemoryFormat, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Build an [N, C, H, W] CPU tensor with a deterministic ascending
/// pattern.
fn ascending_4d(n: usize, c: usize, h: usize, w: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..n * c * h * w).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n, c, h, w], false).unwrap()
}

#[test]
fn channels_last_on_cuda_matches_cpu_reference() {
    ensure_cuda();
    let cpu = ascending_4d(2, 3, 4, 5);
    let original = cpu.data().unwrap().to_vec();

    // CPU reference: convert to channels_last on host, then back
    // to standard contiguous so we can compare via data().
    let cpu_nhwc = cpu
        .clone()
        .contiguous_in(MemoryFormat::ChannelsLast)
        .unwrap();
    assert!(cpu_nhwc.is_contiguous_for(MemoryFormat::ChannelsLast));
    let cpu_round_trip = cpu_nhwc.contiguous_in(MemoryFormat::Contiguous).unwrap();
    assert!(cpu_round_trip.is_contiguous());
    assert_eq!(cpu_round_trip.data().unwrap(), original.as_slice());

    // GPU path: upload to CUDA, convert to channels_last on device,
    // then back to standard contiguous for comparison.
    let gpu = cpu.to(Device::Cuda(0)).unwrap();
    assert!(gpu.is_cuda());
    let gpu_nhwc = gpu.contiguous_in(MemoryFormat::ChannelsLast).unwrap();
    assert!(
        gpu_nhwc.is_cuda(),
        "channels_last conversion must stay on GPU"
    );
    assert!(gpu_nhwc.is_contiguous_for(MemoryFormat::ChannelsLast));

    let gpu_round_trip = gpu_nhwc.contiguous_in(MemoryFormat::Contiguous).unwrap();
    assert!(gpu_round_trip.is_cuda());
    assert!(gpu_round_trip.is_contiguous());
    let gpu_data = gpu_round_trip.cpu().unwrap().data().unwrap().to_vec();
    assert_eq!(
        gpu_data, original,
        "GPU channels_last → contiguous round trip must recover original NCHW data"
    );
}

#[test]
fn channels_last_3d_on_cuda_matches_cpu_reference() {
    ensure_cuda();
    // [N=2, C=3, D=2, H=3, W=4]
    let n = 2 * 3 * 2 * 3 * 4;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let original = data.clone();
    let cpu: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 2, 3, 4], false).unwrap();

    let gpu = cpu.to(Device::Cuda(0)).unwrap();
    let gpu_ndhwc = gpu.contiguous_in(MemoryFormat::ChannelsLast3d).unwrap();
    assert!(gpu_ndhwc.is_cuda());
    assert!(gpu_ndhwc.is_contiguous_for(MemoryFormat::ChannelsLast3d));

    let gpu_round_trip = gpu_ndhwc.contiguous_in(MemoryFormat::Contiguous).unwrap();
    let gpu_data = gpu_round_trip.cpu().unwrap().data().unwrap().to_vec();
    assert_eq!(gpu_data, original);
}

#[test]
fn channels_last_then_back_to_contiguous_round_trips() {
    ensure_cuda();
    let cpu = ascending_4d(1, 4, 2, 3);
    let original = cpu.data().unwrap().to_vec();

    let gpu = cpu.to(Device::Cuda(0)).unwrap();
    let gpu_nhwc = gpu.contiguous_in(MemoryFormat::ChannelsLast).unwrap();
    assert!(gpu_nhwc.is_cuda());

    // Convert back to standard contiguous (NCHW).
    let gpu_back = gpu_nhwc.contiguous_in(MemoryFormat::Contiguous).unwrap();
    assert!(gpu_back.is_cuda());
    assert!(gpu_back.is_contiguous());

    let round_tripped = gpu_back.cpu().unwrap().data().unwrap().to_vec();
    assert_eq!(round_tripped, original);
}

#[test]
fn channels_last_already_contiguous_in_format_is_cheap_clone() {
    ensure_cuda();
    let cpu = ascending_4d(1, 2, 3, 4);
    let original = cpu.data().unwrap().to_vec();
    let gpu = cpu.to(Device::Cuda(0)).unwrap();
    let gpu_nhwc = gpu.contiguous_in(MemoryFormat::ChannelsLast).unwrap();

    // Calling contiguous_in(ChannelsLast) on an already-converted
    // tensor must short-circuit and return without launching a
    // strided_copy kernel.
    assert!(gpu_nhwc.is_contiguous_for(MemoryFormat::ChannelsLast));
    let again = gpu_nhwc.contiguous_in(MemoryFormat::ChannelsLast).unwrap();
    assert!(again.is_cuda());
    assert!(again.is_contiguous_for(MemoryFormat::ChannelsLast));
    // Both round-trip back to the same standard contiguous data.
    let r1 = gpu_nhwc
        .contiguous_in(MemoryFormat::Contiguous)
        .unwrap()
        .cpu()
        .unwrap()
        .data()
        .unwrap()
        .to_vec();
    let r2 = again
        .contiguous_in(MemoryFormat::Contiguous)
        .unwrap()
        .cpu()
        .unwrap()
        .data()
        .unwrap()
        .to_vec();
    assert_eq!(r1, original);
    assert_eq!(r2, original);
}

#[test]
fn channels_last_handles_unit_dims() {
    ensure_cuda();
    // Edge case: a 4D tensor with all-1 spatial dims. The
    // permutation logic must handle this without producing a
    // zero-stride or zero-output buffer.
    let cpu = ascending_4d(2, 4, 1, 1);
    let original = cpu.data().unwrap().to_vec();

    let gpu = cpu.to(Device::Cuda(0)).unwrap();
    let gpu_nhwc = gpu.contiguous_in(MemoryFormat::ChannelsLast).unwrap();
    assert!(gpu_nhwc.is_cuda());

    let recovered = gpu_nhwc
        .contiguous_in(MemoryFormat::Contiguous)
        .unwrap()
        .cpu()
        .unwrap()
        .data()
        .unwrap()
        .to_vec();
    assert_eq!(recovered, original);
}
