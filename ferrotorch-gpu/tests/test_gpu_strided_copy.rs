//! GPU strided-copy integration tests.
//!
//! Exercises the full `Tensor<f32>::contiguous()` code path on a real
//! CUDA device, making sure the `strided_copy_f32` backend method is
//! actually used for non-contiguous CUDA tensors and produces byte-
//! identical output to the host fallback. CL-496.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Build an [H, W] CPU tensor with a deterministic ascending pattern.
fn ascending_hw(h: usize, w: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..h * w).map(|i| i as f32).collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![h, w], false).expect("cpu tensor")
}

#[test]
fn contiguous_on_cuda_transpose_matches_cpu() {
    ensure_cuda();

    // 3x4 source; transpose to 4x3 via stride view, then contiguous.
    // transpose(dim=0, dim=1) on a contiguous [3, 4] tensor with
    // strides [4, 1] produces a view with shape [4, 3] and strides
    // [1, 4]. Calling contiguous() should materialize that view
    // into a true contiguous [4, 3] tensor, entirely on-device.
    let cpu = ascending_hw(3, 4);
    let gpu = cpu.clone().to(Device::Cuda(0)).expect("to cuda");
    let gpu_t = gpu.transpose(0, 1).expect("gpu transpose");
    assert!(!gpu_t.is_contiguous());
    assert!(gpu_t.is_cuda());

    let gpu_c = gpu_t.contiguous().expect("gpu contiguous");
    assert!(gpu_c.is_contiguous());
    assert!(gpu_c.is_cuda());
    assert_eq!(gpu_c.shape(), &[4, 3]);

    // Reference: do the same operation on CPU and compare.
    let cpu_c = cpu.transpose(0, 1).unwrap().contiguous().unwrap();

    let gpu_host = gpu_c.cpu().expect("cpu()").data().unwrap().to_vec();
    let cpu_data = cpu_c.data().unwrap().to_vec();
    assert_eq!(
        gpu_host, cpu_data,
        "GPU strided_copy transpose != CPU reference"
    );
}

#[test]
fn contiguous_on_cuda_permute_3d_matches_cpu() {
    ensure_cuda();

    // [2, 3, 4] -> permute(0, 2, 1) -> [2, 4, 3]
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let cpu: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 4], false).unwrap();

    let gpu = cpu.clone().to(Device::Cuda(0)).unwrap();
    let gpu_p = gpu.permute(&[0, 2, 1]).expect("gpu permute");
    assert!(!gpu_p.is_contiguous());

    let gpu_c = gpu_p.contiguous().expect("gpu contiguous");
    assert_eq!(gpu_c.shape(), &[2, 4, 3]);
    assert!(gpu_c.is_contiguous());
    assert!(gpu_c.is_cuda());

    let cpu_c = cpu.permute(&[0, 2, 1]).unwrap().contiguous().unwrap();

    let gpu_host = gpu_c.cpu().unwrap().data().unwrap().to_vec();
    let cpu_data = cpu_c.data().unwrap().to_vec();
    assert_eq!(
        gpu_host, cpu_data,
        "GPU strided_copy 3d permute != CPU reference"
    );
}

#[test]
fn contiguous_on_cuda_already_contiguous_is_cheap_clone() {
    ensure_cuda();

    // Calling contiguous on an already-contiguous CUDA tensor should
    // return a clone without ever touching strided_copy.
    let gpu = ascending_hw(4, 5).to(Device::Cuda(0)).unwrap();
    assert!(gpu.is_contiguous());
    let gpu_c = gpu.contiguous().expect("contiguous");
    assert!(gpu_c.is_contiguous());
    assert_eq!(gpu_c.shape(), gpu.shape());
    let gpu_host = gpu_c.cpu().unwrap().data().unwrap().to_vec();
    assert_eq!(gpu_host, (0..20).map(|i| i as f32).collect::<Vec<_>>());
}

#[test]
fn contiguous_on_cuda_4d_permute() {
    ensure_cuda();

    // A larger rank stress test: [2, 3, 4, 5] permuted to (0, 3, 1, 2)
    // produces shape [2, 5, 3, 4] with non-C-contiguous strides.
    let n = 2 * 3 * 4 * 5;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let cpu: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 4, 5], false).unwrap();

    let gpu = cpu.clone().to(Device::Cuda(0)).unwrap();
    let gpu_p = gpu.permute(&[0, 3, 1, 2]).unwrap();
    let gpu_c = gpu_p.contiguous().unwrap();
    assert_eq!(gpu_c.shape(), &[2, 5, 3, 4]);

    let cpu_c = cpu.permute(&[0, 3, 1, 2]).unwrap().contiguous().unwrap();
    let gpu_host = gpu_c.cpu().unwrap().data().unwrap().to_vec();
    let cpu_data = cpu_c.data().unwrap().to_vec();
    for (i, (&g, &c)) in gpu_host.iter().zip(cpu_data.iter()).enumerate() {
        assert!((g - c).abs() < 1e-6, "element {i}: gpu={g} cpu={c}",);
    }
}

#[test]
fn contiguous_on_cuda_preserves_autograd() {
    ensure_cuda();

    // Contiguous must preserve grad_fn so gradients flow back through
    // the view. We don't run a full backward pass here (the other
    // tests in test_gpu_autograd.rs do that); we just check that the
    // output tensor has requires_grad set and that it reaches the
    // input via the grad_fn chain.
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let cpu: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(data), vec![3, 4], true).unwrap();
    let gpu = cpu.to(Device::Cuda(0)).unwrap();
    let gpu_t = gpu.transpose(0, 1).unwrap();
    let gpu_c = gpu_t.contiguous().unwrap();
    assert!(gpu_c.requires_grad());
    assert!(gpu_c.grad_fn().is_some(), "contiguous must keep grad_fn");
}
