//! Permanent regression sentinel for P3: `SparseTensor::to_dense_on(Cuda)`
//! and `SparseTensor::from_dense` against a CUDA dense tensor.
//!
//! Pre-fix observable failure:
//!   1. `SparseTensor::to_dense()` (or any caller) materialized to CPU
//!      regardless of context; producing a CUDA dense from a SparseTensor
//!      required `to_dense().to(Device::Cuda(0))` (host detour).
//!   2. `SparseTensor::from_dense(&cuda_tensor, T::zero())` called
//!      `tensor.data()?` which returns `GpuTensorNotAccessible` for a CUDA
//!      tensor; the call bubbled the error.
//!
//! Post-fix:
//!   * `SparseTensor::to_dense_on(Device::Cuda(0))` dispatches to
//!     `cusparseSparseToDense` and returns a CUDA `Tensor` whose values
//!     match the CPU reference within the standard tolerance.
//!   * `SparseTensor::from_dense(&cuda_tensor, T::zero())` dispatches to
//!     `cusparseDenseToSparse_*` for f32/f64 and round-trips correctly.
//!
//! PyTorch parity (rust-gpu-discipline §3): `torch.Tensor.to_dense()` and
//! `torch.Tensor.to_sparse()` keep the result on the input device and
//! dispatch to cuSPARSE on CUDA.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::sparse::SparseTensor;

const F32_TOL: f32 = 1e-5;
const F64_TOL: f64 = 1e-12;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// Standard 4x4 fixture used across the probe.
fn small_sparse_f32() -> SparseTensor<f32> {
    SparseTensor::new(
        vec![
            vec![0, 1],
            vec![0, 3],
            vec![1, 2],
            vec![2, 0],
            vec![2, 3],
            vec![3, 1],
        ],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![4, 4],
    )
    .expect("sparse f32")
}

fn small_sparse_f64() -> SparseTensor<f64> {
    SparseTensor::new(
        vec![
            vec![0, 1],
            vec![0, 3],
            vec![1, 2],
            vec![2, 0],
            vec![2, 3],
            vec![3, 1],
        ],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![4, 4],
    )
    .expect("sparse f64")
}

// ---------------------------------------------------------------------------
// to_dense_on(Cuda(0))
// ---------------------------------------------------------------------------

#[test]
fn p3_to_dense_on_cuda_basic_f32() {
    ensure_cuda_backend();
    let sp = small_sparse_f32();

    // CPU reference.
    let cpu_dense = sp.to_dense().expect("cpu to_dense f32");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = sp
        .to_dense_on(Device::Cuda(0))
        .expect("gpu to_dense_on f32");
    assert!(
        gpu_dense.is_cuda(),
        "to_dense_on(Cuda) output must remain on CUDA"
    );
    assert_eq!(gpu_dense.shape(), &[4, 4]);

    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu->cpu data");
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_TOL,
            "to_dense_on f32 elem {i}: gpu={a} cpu={b}"
        );
    }
}

#[test]
fn p3_to_dense_on_cuda_basic_f64() {
    ensure_cuda_backend();
    let sp = small_sparse_f64();

    let cpu_dense = sp.to_dense().expect("cpu to_dense f64");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = sp
        .to_dense_on(Device::Cuda(0))
        .expect("gpu to_dense_on f64");
    assert!(gpu_dense.is_cuda(), "to_dense_on(Cuda) output must remain on CUDA");

    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu->cpu data");
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_TOL,
            "to_dense_on f64 elem {i}: gpu={a} cpu={b}"
        );
    }
}

#[test]
fn p3_to_dense_on_cuda_empty_f32() {
    ensure_cuda_backend();
    let sp = SparseTensor::<f32>::new(vec![], vec![], vec![3, 5]).expect("empty sparse");
    let gpu_dense = sp
        .to_dense_on(Device::Cuda(0))
        .expect("empty to_dense_on f32");
    assert!(gpu_dense.is_cuda());
    assert_eq!(gpu_dense.shape(), &[3, 5]);
    let cpu = gpu_dense.cpu().expect("cpu");
    let data = cpu.data().expect("data");
    assert!(data.iter().all(|&v| v == 0.0), "empty must be all zeros");
}

#[test]
fn p3_to_dense_on_cuda_zero_row_f32() {
    ensure_cuda_backend();
    // Rows 1..3 have entries; row 0 has none.
    let sp = SparseTensor::<f32>::new(
        vec![vec![1, 2], vec![2, 0], vec![3, 1]],
        vec![3.0, 4.0, 6.0],
        vec![4, 4],
    )
    .expect("zero-row sparse");
    let cpu_dense = sp.to_dense().expect("cpu");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = sp
        .to_dense_on(Device::Cuda(0))
        .expect("zero-row to_dense_on f32");
    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu data");

    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_TOL,
            "zero-row to_dense_on f32 elem {i}: gpu={a} cpu={b}"
        );
    }
    // Row 0 must be all zeros.
    for (i, &v) in gpu_data.iter().take(4).enumerate() {
        assert!(v == 0.0, "zero-row[0,{i}]: got {v}, want 0");
    }
}

#[test]
fn p3_to_dense_on_cuda_large_density_f32() {
    ensure_cuda_backend();

    let m = 256;
    let k = 256;

    // Deterministic ~5% density.
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for i in 0..m {
        for j in 0..k {
            if (i * 7 + j * 13) % 20 == 0 {
                indices.push(vec![i, j]);
                values.push(((i + j) as f32 * 0.01) + 0.5);
            }
        }
    }
    let sp = SparseTensor::<f32>::new(indices, values, vec![m, k]).expect("large sparse");

    let cpu_dense = sp.to_dense().expect("cpu to_dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = sp
        .to_dense_on(Device::Cuda(0))
        .expect("large to_dense_on f32");
    assert!(gpu_dense.is_cuda());
    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu data");

    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_TOL,
            "large to_dense_on f32 elem {i}: gpu={a} cpu={b}"
        );
    }
}

#[test]
fn p3_to_dense_on_cuda_large_density_f64() {
    ensure_cuda_backend();

    let m = 256;
    let k = 256;

    let mut indices = Vec::new();
    let mut values = Vec::new();
    for i in 0..m {
        for j in 0..k {
            if (i * 7 + j * 13) % 20 == 0 {
                indices.push(vec![i, j]);
                values.push(((i + j) as f64 * 0.01) + 0.5);
            }
        }
    }
    let sp = SparseTensor::<f64>::new(indices, values, vec![m, k]).expect("large sparse");

    let cpu_dense = sp.to_dense().expect("cpu to_dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = sp
        .to_dense_on(Device::Cuda(0))
        .expect("large to_dense_on f64");
    assert!(gpu_dense.is_cuda());
    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu data");

    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_TOL,
            "large to_dense_on f64 elem {i}: gpu={a} cpu={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// from_dense (CUDA input)
// ---------------------------------------------------------------------------

#[test]
fn p3_from_dense_cuda_basic_f32() {
    ensure_cuda_backend();

    // A dense matrix with mostly zeros.
    let dense_data: Vec<f32> = vec![
        0.0, 1.0, 0.0, 2.0, // row 0
        0.0, 0.0, 3.0, 0.0, // row 1
        4.0, 0.0, 0.0, 5.0, // row 2
        0.0, 6.0, 0.0, 0.0, // row 3
    ];
    let dense_cpu = from_vec::<f32>(dense_data.clone(), &[4, 4]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    let sp_gpu_path = SparseTensor::<f32>::from_dense(&dense_gpu, 0.0).expect("gpu from_dense f32");
    assert_eq!(sp_gpu_path.shape(), &[4, 4]);
    assert_eq!(sp_gpu_path.nnz(), 6);

    // Round-trip: re-densify (CPU) and compare.
    let re_dense = sp_gpu_path.to_dense().expect("re_dense");
    let re_data = re_dense.data().expect("re data");
    for (i, (&got, &exp)) in re_data.iter().zip(dense_data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < F32_TOL,
            "from_dense round-trip f32 elem {i}: got {got}, exp {exp}"
        );
    }
}

#[test]
fn p3_from_dense_cuda_basic_f64() {
    ensure_cuda_backend();

    let dense_data: Vec<f64> = vec![
        0.0, 1.0, 0.0, 2.0,
        0.0, 0.0, 3.0, 0.0,
        4.0, 0.0, 0.0, 5.0,
        0.0, 6.0, 0.0, 0.0,
    ];
    let dense_cpu = from_vec::<f64>(dense_data.clone(), &[4, 4]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    let sp_gpu_path = SparseTensor::<f64>::from_dense(&dense_gpu, 0.0).expect("gpu from_dense f64");
    assert_eq!(sp_gpu_path.shape(), &[4, 4]);
    assert_eq!(sp_gpu_path.nnz(), 6);

    let re_dense = sp_gpu_path.to_dense().expect("re_dense");
    let re_data = re_dense.data().expect("re data");
    for (i, (&got, &exp)) in re_data.iter().zip(dense_data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < F64_TOL,
            "from_dense round-trip f64 elem {i}: got {got}, exp {exp}"
        );
    }
}

#[test]
fn p3_from_dense_cuda_all_zero_f32() {
    ensure_cuda_backend();
    let dense_data = vec![0.0f32; 12];
    let dense_cpu = from_vec::<f32>(dense_data, &[3, 4]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    let sp = SparseTensor::<f32>::from_dense(&dense_gpu, 0.0).expect("from_dense all zero");
    assert_eq!(sp.shape(), &[3, 4]);
    assert_eq!(sp.nnz(), 0);
}

#[test]
fn p3_from_dense_cuda_round_trip_to_dense_on_f32() {
    // Round-trip through both new GPU paths: dense (CUDA) -> sparse (CPU
    // storage) -> dense (CUDA via to_dense_on). End-to-end never materialises
    // the dense on CPU.
    ensure_cuda_backend();

    let dense_data: Vec<f32> = vec![
        0.0, 1.5, 0.0, -2.0,
        3.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 4.5, 0.0,
        0.0, -6.0, 0.0, 0.5,
    ];
    let dense_cpu = from_vec::<f32>(dense_data.clone(), &[4, 4]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    let sp = SparseTensor::<f32>::from_dense(&dense_gpu, 0.0).expect("from_dense");
    let dense_back = sp
        .to_dense_on(Device::Cuda(0))
        .expect("to_dense_on round-trip");
    assert!(dense_back.is_cuda());

    let dense_back_cpu = dense_back.cpu().expect("cpu");
    let got = dense_back_cpu.data().expect("data");
    for (i, (&a, &exp)) in got.iter().zip(dense_data.iter()).enumerate() {
        assert!(
            (a - exp).abs() < F32_TOL,
            "round-trip f32 elem {i}: got {a}, exp {exp}"
        );
    }
}
