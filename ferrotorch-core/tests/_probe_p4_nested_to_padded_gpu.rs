//! Permanent regression sentinel for P4 of #806: NestedTensor::to_padded
//! and NestedTensor::from_padded GPU paths.
//!
//! Pre-P4, both methods called `tensor.data()?` on every component (or on
//! the padded input), which returns `GpuTensorNotAccessible` for any
//! CUDA-resident tensor. Post-P4:
//!
//! - `to_padded` allocates an output GPU buffer pre-filled with
//!   `pad_value` via `fill_f{32,64}` and scatters each component into
//!   its slot via `strided_scatter_f{32,64}` — composes from the
//!   existing kernels (#802 / CL-496), no new trait surface.
//! - `from_padded` walks `narrow` (zero-copy stride view) +
//!   `.contiguous()` (which routes through the `strided_copy_f{32,64}`
//!   kernel for non-contiguous CUDA views) per component — also
//!   composite, no new kernels.
//!
//! Coverage:
//!   1. 3-component `[(2,4),(3,4),(1,4)]` round-trip on CUDA, ragged_dim=0.
//!   2. ragged_dim=1 round-trip.
//!   3. Single-component (degenerate batch).
//!   4. Non-zero pad value.
//!   5. f64 round-trip.
//!   6. Stride-view (narrow) component flows through the GPU path.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::nested::NestedTensor;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

fn cuda<T>(data: Vec<T>, shape: &[usize]) -> ferrotorch_core::Tensor<T>
where
    T: ferrotorch_core::dtype::Float + 'static,
{
    let cpu = from_vec::<T>(data, shape).expect("cpu tensor");
    cpu.to(Device::Cuda(0)).expect("cpu->gpu")
}

/// 3-component canonical shape from the P4 spec: `[(2,4),(3,4),(1,4)]`,
/// ragged_dim=0, pad_value=0.0. The padded output should be `[3, 3, 4]`,
/// CUDA-resident, with the components scattered into rows [0..2], [0..3],
/// [0..1] of each batch row and zero-padded elsewhere.
#[test]
fn p4_to_padded_3_components_ragged_dim0_f32() {
    ensure_cuda_backend();

    let t1 = cuda::<f32>((1..=8).map(|i| i as f32).collect(), &[2, 4]);
    let t2 = cuda::<f32>((1..=12).map(|i| i as f32 + 10.0).collect(), &[3, 4]);
    let t3 = cuda::<f32>((1..=4).map(|i| i as f32 + 100.0).collect(), &[1, 4]);

    let nt = NestedTensor::new(vec![t1, t2, t3], 0).expect("nested");
    let padded = nt.to_padded(0.0).expect("to_padded");

    assert!(padded.is_cuda(), "P4: to_padded output must remain on CUDA");
    assert_eq!(padded.shape(), &[3, 3, 4]);

    let host = padded.cpu().expect("gpu->cpu");
    let data = host.data().expect("data");

    // batch 0: [(1..8) | zeros(4)]
    assert_eq!(&data[0..8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(&data[8..12], &[0.0; 4]);

    // batch 1: [(11..22)]  (3 rows × 4 cols, fully populated)
    assert_eq!(&data[12..24], &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]);

    // batch 2: [(101..104) | zeros(8)]
    assert_eq!(&data[24..28], &[101.0, 102.0, 103.0, 104.0]);
    assert_eq!(&data[28..36], &[0.0; 8]);
}

/// Round-trip: from_padded of the to_padded output, with the right
/// lengths, recovers each component's values (still on CUDA).
#[test]
fn p4_from_padded_round_trip_ragged_dim0_f32() {
    ensure_cuda_backend();

    let t1 = cuda::<f32>((1..=8).map(|i| i as f32).collect(), &[2, 4]);
    let t2 = cuda::<f32>((1..=12).map(|i| i as f32 + 10.0).collect(), &[3, 4]);
    let t3 = cuda::<f32>((1..=4).map(|i| i as f32 + 100.0).collect(), &[1, 4]);

    let nt = NestedTensor::new(vec![t1, t2, t3], 0).expect("nested");
    let padded = nt.to_padded(0.0).expect("to_padded");
    let lengths = nt.ragged_lengths();

    let recovered = NestedTensor::<f32>::from_padded(&padded, &lengths, 0).expect("from_padded");
    assert_eq!(recovered.num_components(), 3);
    for comp in recovered.tensors() {
        assert!(comp.is_cuda(), "P4: from_padded components must be on CUDA");
    }
    assert_eq!(recovered.tensors()[0].shape(), &[2, 4]);
    assert_eq!(recovered.tensors()[1].shape(), &[3, 4]);
    assert_eq!(recovered.tensors()[2].shape(), &[1, 4]);

    let r0 = recovered.tensors()[0].cpu().unwrap();
    let r1 = recovered.tensors()[1].cpu().unwrap();
    let r2 = recovered.tensors()[2].cpu().unwrap();
    assert_eq!(r0.data().unwrap(), &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        r1.data().unwrap(),
        &[11.0_f32, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
    );
    assert_eq!(r2.data().unwrap(), &[101.0_f32, 102.0, 103.0, 104.0]);
}

/// ragged_dim=1: components have shape `[2, L_i]`, padded to `[batch, 2,
/// max_L]`. Verifies that strided_scatter computes the right slot offset
/// when ragged isn't the leading axis.
#[test]
fn p4_to_padded_ragged_dim1_f32() {
    ensure_cuda_backend();

    // Component 0: shape [2, 3]
    let t1 = cuda::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // Component 1: shape [2, 2]
    let t2 = cuda::<f32>(vec![7.0, 8.0, 9.0, 10.0], &[2, 2]);

    let nt = NestedTensor::new(vec![t1, t2], 1).expect("nested");
    let padded = nt.to_padded(-1.0).expect("to_padded");

    assert!(padded.is_cuda());
    assert_eq!(padded.shape(), &[2, 2, 3]);

    let host = padded.cpu().unwrap();
    let data = host.data().unwrap();

    // batch 0: [[1,2,3],[4,5,6]] (no padding — already at max)
    assert_eq!(&data[0..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // batch 1: [[7,8,-1],[9,10,-1]]
    assert_eq!(&data[6..12], &[7.0, 8.0, -1.0, 9.0, 10.0, -1.0]);

    // Round-trip recovers components.
    let lengths = nt.ragged_lengths();
    let recovered = NestedTensor::<f32>::from_padded(&padded, &lengths, 1).expect("from_padded");
    assert_eq!(recovered.num_components(), 2);
    let r0 = recovered.tensors()[0].cpu().unwrap();
    let r1 = recovered.tensors()[1].cpu().unwrap();
    assert!(recovered.tensors()[0].is_cuda());
    assert!(recovered.tensors()[1].is_cuda());
    assert_eq!(r0.shape(), &[2, 3]);
    assert_eq!(r1.shape(), &[2, 2]);
    assert_eq!(r0.data().unwrap(), &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(r1.data().unwrap(), &[7.0_f32, 8.0, 9.0, 10.0]);
}

/// Single-component edge: degenerate batch of one component.
#[test]
fn p4_single_component_f32() {
    ensure_cuda_backend();

    let t = cuda::<f32>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let nt = NestedTensor::new(vec![t], 0).expect("nested");

    let padded = nt.to_padded(0.0).expect("to_padded");
    assert!(padded.is_cuda());
    assert_eq!(padded.shape(), &[1, 3, 2]);

    let host = padded.cpu().unwrap();
    assert_eq!(host.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let recovered =
        NestedTensor::<f32>::from_padded(&padded, &nt.ragged_lengths(), 0).expect("from_padded");
    assert_eq!(recovered.num_components(), 1);
    assert!(recovered.tensors()[0].is_cuda());
    let r0 = recovered.tensors()[0].cpu().unwrap();
    assert_eq!(r0.data().unwrap(), &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

/// Non-zero padding value must propagate to every padded slot.
#[test]
fn p4_nonzero_pad_value_f32() {
    ensure_cuda_backend();

    let t1 = cuda::<f32>(vec![1.0, 2.0], &[1, 2]);
    let t2 = cuda::<f32>(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[3, 2]);
    let nt = NestedTensor::new(vec![t1, t2], 0).expect("nested");

    let padded = nt.to_padded(-99.5).expect("to_padded");
    assert_eq!(padded.shape(), &[2, 3, 2]);
    let host = padded.cpu().unwrap();
    let data = host.data().unwrap();

    // batch 0: [[1,2], [-99.5, -99.5], [-99.5, -99.5]]
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 2.0);
    for &v in &data[2..6] {
        assert!((v - -99.5).abs() < 1e-6, "expected -99.5, got {v}");
    }
    // batch 1: full
    assert_eq!(&data[6..12], &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

/// f64 path: same kernel surface as f32 (`fill_f64`, `strided_scatter_f64`,
/// `strided_copy_f64`), exercised here to ensure the f64 PTX templates
/// route correctly.
#[test]
fn p4_f64_round_trip() {
    ensure_cuda_backend();

    let t1 = cuda::<f64>(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = cuda::<f64>(vec![5.0_f64, 6.0], &[1, 2]);
    let nt = NestedTensor::new(vec![t1, t2], 0).expect("nested");

    let padded = nt.to_padded(0.0_f64).expect("to_padded");
    assert!(padded.is_cuda());
    assert_eq!(padded.shape(), &[2, 2, 2]);

    let host = padded.cpu().unwrap();
    let data = host.data().unwrap();
    assert_eq!(&data[0..4], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&data[4..6], &[5.0, 6.0]);
    assert_eq!(&data[6..8], &[0.0, 0.0]);

    let recovered =
        NestedTensor::<f64>::from_padded(&padded, &nt.ragged_lengths(), 0).expect("from_padded");
    assert!(recovered.tensors()[0].is_cuda());
    assert!(recovered.tensors()[1].is_cuda());
    let r0 = recovered.tensors()[0].cpu().unwrap();
    let r1 = recovered.tensors()[1].cpu().unwrap();
    assert_eq!(r0.data().unwrap(), &[1.0_f64, 2.0, 3.0, 4.0]);
    assert_eq!(r1.data().unwrap(), &[5.0_f64, 6.0]);
}

/// Stride-view component (a `narrow`) must flow through the GPU path —
/// `try_to_padded_gpu` calls `.contiguous()` on each component so a
/// non-contiguous CUDA view dispatches to `strided_copy_*` and stays on
/// device.
#[test]
fn p4_stride_view_component_stays_on_gpu_f32() {
    ensure_cuda_backend();

    // Build a 4×4 source on GPU, narrow to a 2×4 view, then put it as
    // the first component. This exercises the contiguous() rewrite
    // inside the GPU fast path.
    let src = cuda::<f32>((0..16).map(|i| i as f32).collect(), &[4, 4]);
    let view = src.narrow(0, 1, 2).expect("narrow rows"); // shape [2, 4], non-contig storage
    let t2 = cuda::<f32>((100..104).map(|i| i as f32).collect(), &[1, 4]);

    let nt = NestedTensor::new(vec![view, t2], 0).expect("nested");
    let padded = nt.to_padded(0.0).expect("to_padded");
    assert!(padded.is_cuda());
    assert_eq!(padded.shape(), &[2, 2, 4]);

    let host = padded.cpu().unwrap();
    let data = host.data().unwrap();
    // batch 0: rows [1..3] of src, i.e. values 4..12
    assert_eq!(&data[0..8], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    // batch 1: [100,101,102,103] then 0.0 padding
    assert_eq!(&data[8..12], &[100.0, 101.0, 102.0, 103.0]);
    assert_eq!(&data[12..16], &[0.0; 4]);
}

/// Mismatched-device components surface as DeviceMismatch from the GPU
/// pre-flight in `try_to_padded_gpu`.
#[test]
fn p4_mixed_device_components_rejected() {
    ensure_cuda_backend();

    let cuda_t = cuda::<f32>(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let cpu_t = from_vec::<f32>(vec![5.0, 6.0], &[1, 2]).unwrap();
    let nt = NestedTensor::new(vec![cuda_t, cpu_t], 0).expect("nested");

    // First component is CUDA, second is CPU. The pre-flight check sees
    // the first as CUDA and the second as Device::Cpu — falls through
    // to the CPU path. The CPU path then errors at `tensor.data()?`
    // when it hits the CUDA component (GpuTensorNotAccessible).
    let result = nt.to_padded(0.0);
    assert!(
        result.is_err(),
        "to_padded with mixed devices must error rather than silently corrupt"
    );
}
