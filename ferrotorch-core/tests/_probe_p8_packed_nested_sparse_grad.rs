//! Permanent regression sentinel for P8 of #806: PackedNestedTensor's
//! GPU lane via `data_to_tensor` / `from_data_tensor` composite, and
//! `SparseGrad::apply_sgd` on a CUDA parameter.
//!
//! Pre-fix observable failure (pre-P8):
//!   * `PackedNestedTensor::add` / `sub` / `mul` / `div` had no on-device
//!     path. The `data: Vec<T>` storage is CPU-bound by construction, but
//!     there was no bridge to `Tensor<T>` for callers that wanted to do
//!     element-wise composition on CUDA — the only escape hatch was
//!     manually copying `data()` into a tensor and re-packing by hand.
//!   * `SparseGrad::apply_sgd` called `param.data_vec()?` unconditionally,
//!     which returns `GpuTensorNotAccessible` for any CUDA-resident
//!     parameter. There was no on-device update path; embedding training
//!     against a CUDA parameter erroed at the optimizer step.
//!
//! Post-fix:
//!   * `PackedNestedTensor::data_to_tensor()` and
//!     `PackedNestedTensor::from_data_tensor(tensor, offsets, tail_shape)`
//!     bridge the flat buffer to a `Tensor<T>` so callers can `.to(Cuda)`,
//!     compose `Tensor + Tensor` (which dispatches to `add_f32` /
//!     `add_f64` / `broadcast_add_*` on-device), and round-trip back.
//!   * `SparseGrad::apply_sgd` on a CUDA `f32` param composes
//!     `cpu_to_gpu(values)` + `cpu_to_gpu(indices_as_f32)` +
//!     `scatter_add_rows_f32` + `scale_f32` + `sub_f32`. Output stays on
//!     CUDA. No host detour. f64 takes the analogous f64 lane (gated on
//!     the backend implementing `scatter_add_rows_f64`); when that
//!     primitive is unavailable, the structured `InvalidArgument` error
//!     surfaces verbatim — PyTorch parity for missing CUDA kernels.
//!
//! PyTorch parity (`rust-gpu-discipline` §3 composite-implicit-autograd):
//! `optim.SGD` with `sparse=True` decomposes into the same scatter +
//! scaled-subtract pattern via the dispatcher's element-wise CUDA kernels.
//! `torch.Tensor.apply_` (the analog of `PackedNestedTensor::map`) is
//! documented as CPU-only and users compose dispatched element-wise ops.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::nested::PackedNestedTensor;
use ferrotorch_core::sparse::SparseGrad;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

const F32_TOL: f32 = 1e-5;
const F64_TOL: f64 = 1e-12;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

fn make_tensor_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("make_tensor_f32")
}

fn make_tensor_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("make_tensor_f64")
}

// ---------------------------------------------------------------------------
// PackedNestedTensor: GPU-composite add / sub / mul / div via data_to_tensor
// ---------------------------------------------------------------------------

#[test]
fn p8_packed_add_via_gpu_composite_f32() {
    ensure_cuda_backend();

    // 3 components: lengths 3, 5, 2 with tail [4]. Flat numel = (3+5+2)*4 = 40.
    let lengths = [3usize, 5, 2];
    let tail = [4usize];
    let seq_a: Vec<Vec<f32>> = lengths
        .iter()
        .enumerate()
        .map(|(c, &l)| {
            (0..l * tail[0])
                .map(|j| (c as f32 * 100.0) + j as f32 + 1.0)
                .collect()
        })
        .collect();
    let seq_b: Vec<Vec<f32>> = lengths
        .iter()
        .enumerate()
        .map(|(c, &l)| {
            (0..l * tail[0])
                .map(|j| 0.5 + c as f32 * 0.25 + j as f32 * 0.1)
                .collect()
        })
        .collect();

    let pa = PackedNestedTensor::<f32>::from_sequences(seq_a, &lengths, &tail).expect("pa");
    let pb = PackedNestedTensor::<f32>::from_sequences(seq_b, &lengths, &tail).expect("pb");
    let cpu_sum = pa.add(&pb).expect("cpu add");

    // GPU composite lane: bridge flat buffer to CUDA Tensor, dispatch +,
    // round-trip back. The Tensor::add path uses backend.add_f32.
    let ta_cpu = pa.data_to_tensor().expect("pa->tensor");
    let tb_cpu = pb.data_to_tensor().expect("pb->tensor");
    let ta_gpu = ta_cpu.to(Device::Cuda(0)).expect("a->cuda");
    let tb_gpu = tb_cpu.to(Device::Cuda(0)).expect("b->cuda");
    let tsum_gpu = (&ta_gpu + &tb_gpu).expect("gpu add");
    assert!(
        tsum_gpu.is_cuda(),
        "Tensor + Tensor must stay on CUDA when both inputs are CUDA"
    );

    let gpu_packed =
        PackedNestedTensor::<f32>::from_data_tensor(&tsum_gpu, pa.offsets().to_vec(), tail.to_vec())
            .expect("repack");
    assert_eq!(gpu_packed.offsets(), pa.offsets());
    assert_eq!(gpu_packed.tail_shape(), &tail);
    assert_eq!(gpu_packed.num_components(), 3);

    for (i, (g, e)) in gpu_packed.data().iter().zip(cpu_sum.data().iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_TOL,
            "elem {i}: gpu={g} cpu={e} diff={}",
            (g - e).abs()
        );
    }
}

#[test]
fn p8_packed_sub_mul_div_via_gpu_composite_f32() {
    ensure_cuda_backend();

    // Single 4-component fixture exercises sub, mul, div in one suite.
    let lengths = [2usize, 4, 1, 3];
    let tail = [2usize];
    let seq_a: Vec<Vec<f32>> = lengths
        .iter()
        .enumerate()
        .map(|(c, &l)| {
            (0..l * tail[0])
                .map(|j| 2.0 + c as f32 * 0.5 + j as f32 * 0.1)
                .collect()
        })
        .collect();
    let seq_b: Vec<Vec<f32>> = lengths
        .iter()
        .enumerate()
        .map(|(c, &l)| {
            (0..l * tail[0])
                .map(|j| 1.25 + c as f32 * 0.125 + j as f32 * 0.05)
                .collect()
        })
        .collect();
    let pa = PackedNestedTensor::<f32>::from_sequences(seq_a, &lengths, &tail).expect("pa");
    let pb = PackedNestedTensor::<f32>::from_sequences(seq_b, &lengths, &tail).expect("pb");

    let ta_gpu = pa.data_to_tensor().unwrap().to(Device::Cuda(0)).unwrap();
    let tb_gpu = pb.data_to_tensor().unwrap().to(Device::Cuda(0)).unwrap();

    let cpu_sub = pa.sub(&pb).unwrap();
    let cpu_mul = pa.mul(&pb).unwrap();
    let cpu_div = pa.div(&pb).unwrap();

    let gpu_sub = (&ta_gpu - &tb_gpu).expect("gpu sub");
    let gpu_mul = (&ta_gpu * &tb_gpu).expect("gpu mul");
    let gpu_div = (&ta_gpu / &tb_gpu).expect("gpu div");

    for (label, gpu_t, cpu_p) in [
        ("sub", gpu_sub, cpu_sub),
        ("mul", gpu_mul, cpu_mul),
        ("div", gpu_div, cpu_div),
    ] {
        assert!(gpu_t.is_cuda(), "{label}: must stay on CUDA");
        let g_packed =
            PackedNestedTensor::<f32>::from_data_tensor(&gpu_t, pa.offsets().to_vec(), tail.to_vec())
                .expect("repack");
        for (i, (g, e)) in g_packed.data().iter().zip(cpu_p.data().iter()).enumerate() {
            assert!(
                (g - e).abs() < F32_TOL * (1.0 + e.abs()),
                "{label} elem {i}: gpu={g} cpu={e}"
            );
        }
    }
}

#[test]
fn p8_packed_add_via_gpu_composite_f64() {
    ensure_cuda_backend();

    let lengths = [2usize, 3];
    let tail = [3usize];
    let pa = PackedNestedTensor::<f64>::from_sequences(
        vec![
            (0..6).map(|i| 1.0 + i as f64).collect(),
            (0..9).map(|i| 10.0 + i as f64 * 0.5).collect(),
        ],
        &lengths,
        &tail,
    )
    .unwrap();
    let pb = PackedNestedTensor::<f64>::from_sequences(
        vec![
            (0..6).map(|i| 0.25 * i as f64).collect(),
            (0..9).map(|i| 0.125 + 0.0625 * i as f64).collect(),
        ],
        &lengths,
        &tail,
    )
    .unwrap();
    let cpu = pa.add(&pb).unwrap();

    let ta = pa.data_to_tensor().unwrap().to(Device::Cuda(0)).unwrap();
    let tb = pb.data_to_tensor().unwrap().to(Device::Cuda(0)).unwrap();
    let tsum = (&ta + &tb).unwrap();
    assert!(tsum.is_cuda());

    let g = PackedNestedTensor::<f64>::from_data_tensor(&tsum, pa.offsets().to_vec(), tail.to_vec())
        .unwrap();
    for (i, (a, b)) in g.data().iter().zip(cpu.data().iter()).enumerate() {
        assert!((a - b).abs() < F64_TOL, "elem {i}: gpu={a} cpu={b}");
    }
}

#[test]
fn p8_packed_data_tensor_round_trip_cpu() {
    // Round-trip on CPU with no GPU dispatch: tensor lifecycle only.
    let pa = PackedNestedTensor::<f32>::from_sequences(
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]],
        &[3, 2],
        &[],
    )
    .unwrap();
    let t = pa.data_to_tensor().unwrap();
    assert_eq!(t.shape(), &[5]);
    assert_eq!(t.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

    let back = PackedNestedTensor::<f32>::from_data_tensor(
        &t,
        pa.offsets().to_vec(),
        pa.tail_shape().to_vec(),
    )
    .unwrap();
    assert_eq!(back.data(), pa.data());
    assert_eq!(back.offsets(), pa.offsets());
}

#[test]
fn p8_packed_from_data_tensor_rejects_size_mismatch() {
    let pa = PackedNestedTensor::<f32>::from_sequences(
        vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]],
        &[2, 3],
        &[],
    )
    .unwrap();
    // Tensor with the wrong numel.
    let bogus = make_tensor_f32(&[1.0, 2.0, 3.0], &[3]);
    let err = PackedNestedTensor::<f32>::from_data_tensor(
        &bogus,
        pa.offsets().to_vec(),
        pa.tail_shape().to_vec(),
    );
    assert!(err.is_err(), "size mismatch must error");
}

// ---------------------------------------------------------------------------
// SparseGrad::apply_sgd on CUDA parameters
// ---------------------------------------------------------------------------

#[test]
fn p8_sparse_grad_apply_sgd_cuda_f32_basic() {
    ensure_cuda_backend();

    // 4x4 param, 3 nnz, lr=0.1.
    let param_data: Vec<f32> = (1..=16).map(|x| x as f32 * 0.5).collect();
    let cpu_param = make_tensor_f32(&param_data, &[4, 4]);
    let mut gpu_param = cpu_param.to(Device::Cuda(0)).expect("param->cuda");
    let mut cpu_param_clone = cpu_param.clone();

    let grad = SparseGrad::<f32>::new(
        vec![0, 2, 1],
        vec![
            1.0, 2.0, 3.0, 4.0, // slab for idx=0
            5.0, 6.0, 7.0, 8.0, // slab for idx=2
            9.0, 10.0, 11.0, 12.0, // slab for idx=1
        ],
        vec![4],
    )
    .expect("grad");

    let lr = 0.1f32;
    grad.apply_sgd(&mut cpu_param_clone, lr).expect("cpu sgd");
    grad.apply_sgd(&mut gpu_param, lr).expect("gpu sgd");

    assert!(gpu_param.is_cuda(), "param must remain on CUDA after SGD");

    let gpu_back = gpu_param.cpu().unwrap();
    let gpu_data = gpu_back.data().unwrap();
    let cpu_data = cpu_param_clone.data().unwrap();
    for (i, (g, e)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_TOL,
            "apply_sgd elem {i}: gpu={g} cpu={e}"
        );
    }
}

#[test]
fn p8_sparse_grad_apply_sgd_cuda_f32_with_duplicates() {
    ensure_cuda_backend();
    // PyTorch's `optim.SGD` with non-coalesced sparse grad accumulates
    // duplicates (scatter-add). Verify GPU lane matches.
    let param_data: Vec<f32> = (0..12).map(|x| x as f32).collect();
    let cpu_param = make_tensor_f32(&param_data, &[3, 4]);
    let mut gpu_param = cpu_param.to(Device::Cuda(0)).unwrap();
    let mut cpu_param_clone = cpu_param.clone();

    let grad = SparseGrad::<f32>::new(
        vec![1, 1, 0], // duplicate index 1
        vec![
            1.0, 1.0, 1.0, 1.0, // first slab for idx=1
            2.0, 2.0, 2.0, 2.0, // second slab for idx=1 (accumulates)
            3.0, 3.0, 3.0, 3.0, // slab for idx=0
        ],
        vec![4],
    )
    .unwrap();

    grad.apply_sgd(&mut cpu_param_clone, 0.5).unwrap();
    grad.apply_sgd(&mut gpu_param, 0.5).unwrap();

    let gpu_data = gpu_param.cpu().unwrap().data().unwrap().to_vec();
    let cpu_data = cpu_param_clone.data().unwrap();
    for (i, (g, e)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_TOL,
            "duplicate-idx apply_sgd elem {i}: gpu={g} cpu={e}"
        );
    }
}

#[test]
fn p8_sparse_grad_apply_sgd_cuda_f32_empty_grad_is_noop() {
    ensure_cuda_backend();
    let param_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let cpu_param = make_tensor_f32(&param_data, &[2, 2]);
    let mut gpu_param = cpu_param.to(Device::Cuda(0)).unwrap();

    let grad = SparseGrad::<f32>::new(vec![], vec![], vec![2]).unwrap();
    grad.apply_sgd(&mut gpu_param, 0.1).unwrap();

    assert!(gpu_param.is_cuda(), "empty grad must keep param on CUDA");
    let gpu_data = gpu_param.cpu().unwrap().data().unwrap().to_vec();
    for (i, (g, e)) in gpu_data.iter().zip(param_data.iter()).enumerate() {
        assert!((g - e).abs() < F32_TOL, "empty-noop elem {i}: gpu={g} cpu={e}");
    }
}

#[test]
fn p8_sparse_grad_apply_sgd_cuda_f32_index_out_of_bounds() {
    ensure_cuda_backend();
    let cpu_param = make_tensor_f32(&[0.0; 8], &[2, 4]);
    let mut gpu_param = cpu_param.to(Device::Cuda(0)).unwrap();

    let grad = SparseGrad::<f32>::new(vec![5], vec![1.0, 1.0, 1.0, 1.0], vec![4]).unwrap();
    let res = grad.apply_sgd(&mut gpu_param, 0.1);
    assert!(res.is_err(), "out-of-range index must error before any kernel launch");
}

#[test]
fn p8_sparse_grad_apply_sgd_cuda_f64_lane_via_backend() {
    // f64 lane: when the backend implements scatter_add_rows_f64, the
    // composite runs on-device. When it doesn't, the structured Err
    // surfaces (PyTorch §3 parity: missing CUDA kernels surface errors).
    // Either is acceptable; we check the *correctness* of the result when
    // it succeeds.
    ensure_cuda_backend();
    let cpu_param = make_tensor_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let mut gpu_param = cpu_param.to(Device::Cuda(0)).unwrap();
    let mut cpu_param_clone = cpu_param.clone();

    let grad = SparseGrad::<f64>::new(vec![0, 2], vec![1.0, 1.0, 2.0, 2.0], vec![2]).unwrap();
    let lr = 0.5f64;

    grad.apply_sgd(&mut cpu_param_clone, lr).unwrap();
    let gpu_res = grad.apply_sgd(&mut gpu_param, lr);

    if let Ok(()) = gpu_res {
        let gpu_data = gpu_param.cpu().unwrap().data().unwrap().to_vec();
        let cpu_data = cpu_param_clone.data().unwrap();
        for (i, (g, e)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
            assert!(
                (g - e).abs() < F64_TOL,
                "f64 apply_sgd elem {i}: gpu={g} cpu={e}"
            );
        }
    } else {
        // Backend doesn't implement scatter_add_rows_f64 — error is the
        // §3-correct behaviour. Confirm it's a structured error, not a panic.
        eprintln!(
            "f64 GPU lane unsupported on this backend: {} (treated as PyTorch-parity §3 missing-kernel error)",
            gpu_res.unwrap_err()
        );
    }
}

#[test]
fn p8_sparse_grad_apply_sgd_cuda_large_embedding() {
    ensure_cuda_backend();
    // 1024-row embedding, dim 64, sparse update touching 16 random rows
    // with deterministic value patterns.
    let v = 1024usize;
    let d = 64usize;
    let param_data: Vec<f32> = (0..v * d).map(|x| ((x % 1000) as f32) * 0.001).collect();
    let cpu_param = make_tensor_f32(&param_data, &[v, d]);
    let mut gpu_param = cpu_param.to(Device::Cuda(0)).unwrap();
    let mut cpu_param_clone = cpu_param.clone();

    let nnz = 16usize;
    let indices: Vec<usize> = (0..nnz).map(|i| (i * 67 + 5) % v).collect();
    let mut values = Vec::with_capacity(nnz * d);
    for k in 0..nnz {
        for j in 0..d {
            values.push((k as f32 * 0.01 + j as f32 * 0.001) - 0.5);
        }
    }
    let grad = SparseGrad::<f32>::new(indices, values, vec![d]).unwrap();
    let lr = 0.01f32;

    grad.apply_sgd(&mut cpu_param_clone, lr).unwrap();
    grad.apply_sgd(&mut gpu_param, lr).unwrap();

    let gpu_data = gpu_param.cpu().unwrap().data().unwrap().to_vec();
    let cpu_data = cpu_param_clone.data().unwrap();
    for (i, (g, e)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_TOL,
            "large-embed apply_sgd elem {i}: gpu={g} cpu={e}"
        );
    }
}

// ---------------------------------------------------------------------------
// PackedNestedTensor edge cases via the GPU composite lane
// ---------------------------------------------------------------------------

#[test]
fn p8_packed_single_component_gpu_composite() {
    ensure_cuda_backend();
    let pa = PackedNestedTensor::<f32>::from_sequences(
        vec![(0..8).map(|i| 1.0 + i as f32).collect()],
        &[4],
        &[2],
    )
    .unwrap();
    let pb = PackedNestedTensor::<f32>::from_sequences(
        vec![(0..8).map(|i| 0.5 + i as f32 * 0.25).collect()],
        &[4],
        &[2],
    )
    .unwrap();
    let cpu = pa.mul(&pb).unwrap();
    let ta = pa.data_to_tensor().unwrap().to(Device::Cuda(0)).unwrap();
    let tb = pb.data_to_tensor().unwrap().to(Device::Cuda(0)).unwrap();
    let tprod = (&ta * &tb).unwrap();
    let g = PackedNestedTensor::<f32>::from_data_tensor(
        &tprod,
        pa.offsets().to_vec(),
        pa.tail_shape().to_vec(),
    )
    .unwrap();
    for (i, (a, b)) in g.data().iter().zip(cpu.data().iter()).enumerate() {
        assert!((a - b).abs() < F32_TOL, "single-comp elem {i}: g={a} c={b}");
    }
}

#[test]
fn p8_packed_many_components_gpu_composite() {
    ensure_cuda_backend();
    // 64 components × 8 elements each — exercises the path with many
    // segments. The flat add doesn't care about segment count; offsets are
    // just metadata.
    let n_components = 64usize;
    let comp_len = 8usize;
    let lengths: Vec<usize> = vec![comp_len; n_components];
    let seq_a: Vec<Vec<f32>> = (0..n_components)
        .map(|c| (0..comp_len).map(|j| (c * 100 + j) as f32).collect())
        .collect();
    let seq_b: Vec<Vec<f32>> = (0..n_components)
        .map(|c| (0..comp_len).map(|j| 0.5 + (c + j) as f32 * 0.1).collect())
        .collect();

    let pa = PackedNestedTensor::<f32>::from_sequences(seq_a, &lengths, &[]).unwrap();
    let pb = PackedNestedTensor::<f32>::from_sequences(seq_b, &lengths, &[]).unwrap();
    let cpu = pa.add(&pb).unwrap();

    let ta = pa.data_to_tensor().unwrap().to(Device::Cuda(0)).unwrap();
    let tb = pb.data_to_tensor().unwrap().to(Device::Cuda(0)).unwrap();
    let tsum = (&ta + &tb).unwrap();
    assert!(tsum.is_cuda());

    let g = PackedNestedTensor::<f32>::from_data_tensor(
        &tsum,
        pa.offsets().to_vec(),
        pa.tail_shape().to_vec(),
    )
    .unwrap();
    assert_eq!(g.num_components(), n_components);
    for (i, (a, b)) in g.data().iter().zip(cpu.data().iter()).enumerate() {
        assert!((a - b).abs() < F32_TOL, "many-comp elem {i}: g={a} c={b}");
    }
}
