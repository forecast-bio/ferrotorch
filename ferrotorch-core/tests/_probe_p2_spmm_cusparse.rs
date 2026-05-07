//! Permanent regression sentinel for P2: `SparseTensor::spmm` on CUDA.
//!
//! Pre-fix observable failure:
//!   1. Construct a small `SparseTensor` (CPU-side COO storage; that is
//!      intentional — the sparse storage stays CPU-resident in P2 scope).
//!   2. Construct a 2-D dense `Tensor<f32>`/`Tensor<f64>` and move it to
//!      `Device::Cuda(0)`.
//!   3. Call `sparse.spmm(&dense)`. Pre-fix the body called `dense.data()?`
//!      which returns `Err(FerrotorchError::GpuTensorNotAccessible)` for a
//!      CUDA tensor — the spmm bubbled that error.
//!
//! Post-fix the spmm dispatches via `cusparseSpMM` and returns a CUDA
//! `Tensor` whose values match the CPU reference within `F32_MATMUL_GPU =
//! 1e-3` (f32) or `F64_MATMUL_GPU = 1e-9` (f64). Per `rust-gpu-discipline §3`
//! (PyTorch parity), this mirrors `torch.sparse.mm` running on cuSPARSE
//! when the dense operand is CUDA.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::sparse::SparseTensor;

const F32_MATMUL_GPU: f32 = 1e-3;
const F64_MATMUL_GPU: f64 = 1e-9;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// Build the small 4x4 sparse matrix used by the basic-coverage tests.
///
/// Pattern:
///   [ 0, 1, 0, 2 ]
///   [ 0, 0, 3, 0 ]
///   [ 4, 0, 0, 5 ]
///   [ 0, 6, 0, 0 ]
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

/// 4x4 sparse @ 4x3 dense, f32, dense on CUDA.
#[test]
fn p2_spmm_cusparse_basic_f32() {
    ensure_cuda_backend();

    let sp = small_sparse_f32();
    let dense_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let dense_cpu = from_vec::<f32>(dense_data.clone(), &[4, 3]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    // CPU reference (with the dense kept on CPU).
    let cpu_ref = sp.spmm(&dense_cpu).expect("cpu spmm");
    let cpu_data = cpu_ref.data().expect("cpu data").to_vec();

    // GPU path. Pre-fix this errors with GpuTensorNotAccessible from `dense.data()?`.
    let out = sp.spmm(&dense_gpu).expect("cusparse spmm f32");
    assert!(out.is_cuda(), "spmm output must remain on CUDA when input was CUDA");
    assert_eq!(out.shape(), &[4, 3]);

    let out_cpu = out.cpu().expect("out gpu->cpu");
    let out_data = out_cpu.data().expect("out data");
    for (i, (&a, &b)) in out_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_MATMUL_GPU,
            "spmm f32 elem {i}: gpu={a} cpu={b}"
        );
    }
}

/// 4x4 sparse @ 4x3 dense, f64, dense on CUDA.
#[test]
fn p2_spmm_cusparse_basic_f64() {
    ensure_cuda_backend();

    let sp = small_sparse_f64();
    let dense_data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
    let dense_cpu = from_vec::<f64>(dense_data.clone(), &[4, 3]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    let cpu_ref = sp.spmm(&dense_cpu).expect("cpu spmm");
    let cpu_data = cpu_ref.data().expect("cpu data").to_vec();

    let out = sp.spmm(&dense_gpu).expect("cusparse spmm f64");
    assert!(out.is_cuda(), "spmm output must remain on CUDA when input was CUDA");
    assert_eq!(out.shape(), &[4, 3]);

    let out_cpu = out.cpu().expect("out gpu->cpu");
    let out_data = out_cpu.data().expect("out data");
    for (i, (&a, &b)) in out_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_MATMUL_GPU,
            "spmm f64 elem {i}: gpu={a} cpu={b}"
        );
    }
}

/// Empty sparse (nnz=0) — output must be all zeros on CUDA.
#[test]
fn p2_spmm_cusparse_empty_nnz_f32() {
    ensure_cuda_backend();

    let sp = SparseTensor::<f32>::new(vec![], vec![], vec![4, 4]).expect("empty sparse");
    let dense_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let dense_cpu = from_vec::<f32>(dense_data, &[4, 3]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    let out = sp.spmm(&dense_gpu).expect("cusparse spmm empty");
    assert!(out.is_cuda(), "empty-spmm output must be CUDA");
    assert_eq!(out.shape(), &[4, 3]);

    let out_cpu = out.cpu().expect("out gpu->cpu");
    let out_data = out_cpu.data().expect("out data");
    for (i, &v) in out_data.iter().enumerate() {
        assert!(v.abs() < F32_MATMUL_GPU, "empty spmm elem {i}: got {v}, want 0");
    }
}

/// Zero-row sparse — row 0 has no entries; output[0, :] must be zeros.
#[test]
fn p2_spmm_cusparse_zero_row_f32() {
    ensure_cuda_backend();

    // Rows 1..3 have entries; row 0 has none.
    let sp = SparseTensor::<f32>::new(
        vec![vec![1, 2], vec![2, 0], vec![3, 1]],
        vec![3.0, 4.0, 6.0],
        vec![4, 4],
    )
    .expect("zero-row sparse");
    let dense_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let dense_cpu = from_vec::<f32>(dense_data, &[4, 3]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    let cpu_ref = sp.spmm(&dense_cpu).expect("cpu spmm");
    let cpu_data = cpu_ref.data().expect("cpu data").to_vec();

    let out = sp.spmm(&dense_gpu).expect("cusparse spmm zero-row");
    assert!(out.is_cuda(), "zero-row output must be CUDA");
    let out_cpu = out.cpu().expect("out gpu->cpu");
    let out_data = out_cpu.data().expect("out data");

    // Row 0 must be exact zeros.
    for (i, &v) in out_data.iter().take(3).enumerate() {
        assert!(v.abs() < F32_MATMUL_GPU, "zero-row[0,{i}]: got {v}, want 0");
    }
    // Remaining rows match CPU reference.
    for (i, (&a, &b)) in out_data.iter().zip(cpu_data.iter()).enumerate().skip(3) {
        assert!(
            (a - b).abs() < F32_MATMUL_GPU,
            "zero-row spmm elem {i}: gpu={a} cpu={b}"
        );
    }
}

/// Large random-ish 256x256 sparse @ 256x64 dense, ~5% density. f32.
#[test]
fn p2_spmm_cusparse_large_f32() {
    ensure_cuda_backend();

    let m = 256;
    let k = 256;
    let n = 64;

    // Deterministic ~5% density: entry (i, j) when (i*7 + j*13) % 20 == 0.
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

    // Deterministic dense: dense[i, j] = sin(i*0.1 + j*0.2).
    let dense_vec: Vec<f32> = (0..(k * n))
        .map(|idx| {
            let r = idx / n;
            let c = idx % n;
            (r as f32 * 0.1 + c as f32 * 0.2).sin()
        })
        .collect();
    let dense_cpu = from_vec::<f32>(dense_vec, &[k, n]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    let cpu_ref = sp.spmm(&dense_cpu).expect("cpu spmm");
    let cpu_data = cpu_ref.data().expect("cpu data").to_vec();

    let out = sp.spmm(&dense_gpu).expect("cusparse spmm large f32");
    assert!(out.is_cuda(), "large output must be CUDA");
    assert_eq!(out.shape(), &[m, n]);

    let out_cpu = out.cpu().expect("out gpu->cpu");
    let out_data = out_cpu.data().expect("out data");
    for (i, (&a, &b)) in out_data.iter().zip(cpu_data.iter()).enumerate() {
        // Slightly relaxed tolerance for accumulated rounding over ~12.8k nnz.
        assert!(
            (a - b).abs() < 1e-2,
            "large spmm f32 elem {i}: gpu={a} cpu={b}"
        );
    }
}

/// Large random-ish 256x256 sparse @ 256x64 dense, ~5% density. f64.
#[test]
fn p2_spmm_cusparse_large_f64() {
    ensure_cuda_backend();

    let m = 256;
    let k = 256;
    let n = 64;

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

    let dense_vec: Vec<f64> = (0..(k * n))
        .map(|idx| {
            let r = idx / n;
            let c = idx % n;
            (r as f64 * 0.1 + c as f64 * 0.2).sin()
        })
        .collect();
    let dense_cpu = from_vec::<f64>(dense_vec, &[k, n]).expect("dense cpu");
    let dense_gpu = dense_cpu.to(Device::Cuda(0)).expect("dense->gpu");

    let cpu_ref = sp.spmm(&dense_cpu).expect("cpu spmm");
    let cpu_data = cpu_ref.data().expect("cpu data").to_vec();

    let out = sp.spmm(&dense_gpu).expect("cusparse spmm large f64");
    assert!(out.is_cuda(), "large f64 output must be CUDA");
    assert_eq!(out.shape(), &[m, n]);

    let out_cpu = out.cpu().expect("out gpu->cpu");
    let out_data = out_cpu.data().expect("out data");
    for (i, (&a, &b)) in out_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_MATMUL_GPU,
            "large spmm f64 elem {i}: gpu={a} cpu={b}"
        );
    }
}
