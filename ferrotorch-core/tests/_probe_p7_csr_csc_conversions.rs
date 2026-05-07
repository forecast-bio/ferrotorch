//! Permanent regression sentinel for P7: CSR/CSC/COO format conversions and
//! `CscTensor::to_dense` / `CsrTensor::to_dense_on` / `CooTensor::to_dense_on`
//! against CUDA tensors.
//!
//! Pre-fix observable failure (pre-P7):
//!   * `CsrTensor::to_dense()`, `CscTensor::to_dense()`,
//!     `CooTensor::to_dense()` always materialised on CPU regardless of
//!     context — producing a CUDA dense matrix from a CSR/CSC/COO required
//!     `.to_dense().to(Device::Cuda(0))` (host detour).
//!   * Format conversions (`CsrTensor::from_coo`, `CscTensor::from_csr`,
//!     `CscTensor::to_csr`) ran on CPU only; there was no on-device path
//!     even when both inputs and consumers lived on CUDA.
//!
//! Post-fix:
//!   * `to_dense_on(Device::Cuda(0))` dispatches to `cusparseSparseToDense`
//!     with a CSR or CSC descriptor; output stays on device, values match
//!     CPU reference within tolerance.
//!   * `from_coo_on` / `from_csr_on` / `to_csr_on` dispatch to
//!     `cusparseXcoo2csr` / `cusparseCsr2cscEx2` and round-trip COO → CSR →
//!     CSC → dense correctly on CUDA.
//!
//! PyTorch parity (rust-gpu-discipline §3): `torch.sparse_csr_tensor` /
//! `torch.sparse_csc_tensor` / `torch.sparse_coo_tensor` keep results on
//! the input device and the format-conversion helpers dispatch to cuSPARSE
//! on CUDA.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::sparse::{CooTensor, CscTensor, CsrTensor};

const F32_TOL: f32 = 1e-5;
const F64_TOL: f64 = 1e-12;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// 4x4 fixture used across the f32 probes.
fn small_coo_f32() -> CooTensor<f32> {
    // Same nonzero positions as small_sparse_f32 in P3.
    CooTensor::new(
        vec![0, 0, 1, 2, 2, 3],
        vec![1, 3, 2, 0, 3, 1],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        4,
        4,
    )
    .expect("coo f32")
}

fn small_coo_f64() -> CooTensor<f64> {
    CooTensor::new(
        vec![0, 0, 1, 2, 2, 3],
        vec![1, 3, 2, 0, 3, 1],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        4,
        4,
    )
    .expect("coo f64")
}

// ---------------------------------------------------------------------------
// CsrTensor::to_dense_on(Cuda(0))
// ---------------------------------------------------------------------------

#[test]
fn p7_csr_to_dense_on_cuda_basic_f32() {
    ensure_cuda_backend();
    let coo = small_coo_f32();
    let csr = CsrTensor::from_coo(&coo).expect("csr cpu build");

    let cpu_dense = csr.to_dense().expect("cpu csr to_dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = csr
        .to_dense_on(Device::Cuda(0))
        .expect("gpu csr to_dense_on f32");
    assert!(
        gpu_dense.is_cuda(),
        "csr.to_dense_on(Cuda) must stay on CUDA"
    );
    assert_eq!(gpu_dense.shape(), &[4, 4]);

    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu->cpu data");
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_TOL,
            "csr.to_dense_on f32 elem {i}: gpu={a} cpu={b}"
        );
    }
}

#[test]
fn p7_csr_to_dense_on_cuda_basic_f64() {
    ensure_cuda_backend();
    let coo = small_coo_f64();
    let csr = CsrTensor::from_coo(&coo).expect("csr cpu build");

    let cpu_dense = csr.to_dense().expect("cpu csr to_dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = csr
        .to_dense_on(Device::Cuda(0))
        .expect("gpu csr to_dense_on f64");
    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu data");
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_TOL,
            "csr.to_dense_on f64 elem {i}: gpu={a} cpu={b}"
        );
    }
}

#[test]
fn p7_csr_to_dense_on_cuda_empty_f32() {
    ensure_cuda_backend();
    let csr = CsrTensor::<f32>::new(vec![0, 0, 0, 0], vec![], vec![], 3, 5).expect("empty csr");
    let gpu_dense = csr
        .to_dense_on(Device::Cuda(0))
        .expect("empty csr to_dense_on f32");
    assert!(gpu_dense.is_cuda());
    assert_eq!(gpu_dense.shape(), &[3, 5]);
    let cpu = gpu_dense.cpu().expect("cpu");
    let data = cpu.data().expect("data");
    assert!(data.iter().all(|&v| v == 0.0));
}

// ---------------------------------------------------------------------------
// CscTensor::to_dense_on(Cuda(0))
// ---------------------------------------------------------------------------

#[test]
fn p7_csc_to_dense_on_cuda_basic_f32() {
    ensure_cuda_backend();
    let coo = small_coo_f32();
    let csr = CsrTensor::from_coo(&coo).expect("csr");
    let csc = CscTensor::from_csr(&csr);

    let cpu_dense = csc.to_dense().expect("cpu csc to_dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = csc
        .to_dense_on(Device::Cuda(0))
        .expect("gpu csc to_dense_on f32");
    assert!(gpu_dense.is_cuda());
    assert_eq!(gpu_dense.shape(), &[4, 4]);

    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu data");
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_TOL,
            "csc.to_dense_on f32 elem {i}: gpu={a} cpu={b}"
        );
    }
}

#[test]
fn p7_csc_to_dense_on_cuda_basic_f64() {
    ensure_cuda_backend();
    let coo = small_coo_f64();
    let csr = CsrTensor::from_coo(&coo).expect("csr");
    let csc = CscTensor::from_csr(&csr);

    let cpu_dense = csc.to_dense().expect("cpu csc to_dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = csc
        .to_dense_on(Device::Cuda(0))
        .expect("gpu csc to_dense_on f64");
    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu data");
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_TOL,
            "csc.to_dense_on f64 elem {i}: gpu={a} cpu={b}"
        );
    }
}

#[test]
fn p7_csc_to_dense_on_cuda_single_row_f32() {
    ensure_cuda_backend();
    // 1x4 single-row matrix.
    let coo = CooTensor::<f32>::new(
        vec![0, 0],
        vec![1, 3],
        vec![1.5, -2.5],
        1,
        4,
    )
    .expect("coo");
    let csr = CsrTensor::from_coo(&coo).expect("csr");
    let csc = CscTensor::from_csr(&csr);

    let cpu_dense = csc.to_dense().expect("cpu");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = csc
        .to_dense_on(Device::Cuda(0))
        .expect("gpu csc single-row");
    let gpu_data = gpu_dense.cpu().unwrap().data().unwrap().to_vec();
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_TOL,
            "single-row elem {i}: gpu={a} cpu={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// CooTensor::to_dense_on(Cuda(0))
// ---------------------------------------------------------------------------

#[test]
fn p7_coo_to_dense_on_cuda_basic_f32() {
    ensure_cuda_backend();
    let coo = small_coo_f32();

    let cpu_dense = coo.to_dense().expect("cpu coo to_dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = coo
        .to_dense_on(Device::Cuda(0))
        .expect("gpu coo to_dense_on f32");
    assert!(gpu_dense.is_cuda());

    let gpu_back = gpu_dense.cpu().expect("gpu->cpu");
    let gpu_data = gpu_back.data().expect("gpu data");
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_TOL,
            "coo.to_dense_on f32 elem {i}: gpu={a} cpu={b}"
        );
    }
}

#[test]
fn p7_coo_to_dense_on_cuda_basic_f64() {
    ensure_cuda_backend();
    let coo = small_coo_f64();

    let cpu_dense = coo.to_dense().expect("cpu");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_dense = coo.to_dense_on(Device::Cuda(0)).expect("gpu coo f64");
    let gpu_data = gpu_dense.cpu().unwrap().data().unwrap().to_vec();
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_TOL,
            "coo.to_dense_on f64 elem {i}: gpu={a} cpu={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Round-trip: COO → CSR → CSC → dense, all on CUDA.
// ---------------------------------------------------------------------------

#[test]
fn p7_roundtrip_coo_csr_csc_dense_cuda_f32() {
    ensure_cuda_backend();
    let coo = small_coo_f32();

    // CPU oracle.
    let cpu_csr = CsrTensor::from_coo(&coo).expect("cpu csr");
    let cpu_csc = CscTensor::from_csr(&cpu_csr);
    let cpu_dense = cpu_csc.to_dense().expect("cpu dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    // GPU lane.
    let gpu_csr = CsrTensor::from_coo_on(&coo, Device::Cuda(0)).expect("gpu csr build");
    let gpu_csc = CscTensor::from_csr_on(&gpu_csr, Device::Cuda(0)).expect("gpu csc build");
    let gpu_dense = gpu_csc
        .to_dense_on(Device::Cuda(0))
        .expect("gpu dense materialise");
    assert!(gpu_dense.is_cuda());

    let gpu_data = gpu_dense.cpu().unwrap().data().unwrap().to_vec();
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_TOL,
            "roundtrip f32 elem {i}: gpu={a} cpu={b}"
        );
    }

    // CSC → CSR via to_csr_on round-trips back to the original CSR.
    let gpu_csr2 = gpu_csc.to_csr_on(Device::Cuda(0)).expect("gpu csc->csr");
    assert_eq!(gpu_csr2.row_ptrs(), gpu_csr.row_ptrs());
    assert_eq!(gpu_csr2.col_indices(), gpu_csr.col_indices());
}

#[test]
fn p7_roundtrip_coo_csr_csc_dense_cuda_f64() {
    ensure_cuda_backend();
    let coo = small_coo_f64();

    let cpu_csr = CsrTensor::from_coo(&coo).expect("cpu csr");
    let cpu_csc = CscTensor::from_csr(&cpu_csr);
    let cpu_dense = cpu_csc.to_dense().expect("cpu dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_csr = CsrTensor::from_coo_on(&coo, Device::Cuda(0)).expect("gpu csr");
    let gpu_csc = CscTensor::from_csr_on(&gpu_csr, Device::Cuda(0)).expect("gpu csc");
    let gpu_dense = gpu_csc.to_dense_on(Device::Cuda(0)).expect("gpu dense");
    let gpu_data = gpu_dense.cpu().unwrap().data().unwrap().to_vec();
    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_TOL,
            "roundtrip f64 elem {i}: gpu={a} cpu={b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Edge cases: empty, single-row, large 1024×1024 ~1% density.
// ---------------------------------------------------------------------------

#[test]
fn p7_empty_matrix_cuda_f32() {
    ensure_cuda_backend();
    let coo = CooTensor::<f32>::new(vec![], vec![], vec![], 4, 5).expect("empty coo");
    let csr = CsrTensor::from_coo_on(&coo, Device::Cuda(0)).expect("empty csr gpu");
    assert_eq!(csr.nnz(), 0);
    let csc = CscTensor::from_csr_on(&csr, Device::Cuda(0)).expect("empty csc gpu");
    assert_eq!(csc.nnz(), 0);
    let dense = csc.to_dense_on(Device::Cuda(0)).expect("empty dense gpu");
    let data = dense.cpu().unwrap().data().unwrap().to_vec();
    assert_eq!(data.len(), 20);
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn p7_single_row_cuda_f32() {
    ensure_cuda_backend();
    let coo = CooTensor::<f32>::new(
        vec![0, 0, 0],
        vec![0, 2, 4],
        vec![1.0, 2.0, 3.0],
        1,
        5,
    )
    .expect("single-row coo");
    let csr = CsrTensor::from_coo_on(&coo, Device::Cuda(0)).expect("csr gpu");
    let csc = CscTensor::from_csr_on(&csr, Device::Cuda(0)).expect("csc gpu");
    let dense = csc.to_dense_on(Device::Cuda(0)).expect("dense gpu");
    let data = dense.cpu().unwrap().data().unwrap().to_vec();
    assert_eq!(data, vec![1.0, 0.0, 2.0, 0.0, 3.0]);
}

#[test]
fn p7_large_1024_density_1pct_cuda_f32() {
    ensure_cuda_backend();
    let m = 1024usize;
    let n = 1024usize;

    // Deterministic ~1% density.
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if (i * 31 + j * 47) % 100 == 0 {
                rows.push(i);
                cols.push(j);
                vals.push(((i + 2 * j) as f32 * 0.0001) + 0.5);
            }
        }
    }
    let coo = CooTensor::<f32>::new(rows, cols, vals, m, n).expect("large coo");

    // CPU oracle.
    let cpu_csr = CsrTensor::from_coo(&coo).expect("cpu csr");
    let cpu_csc = CscTensor::from_csr(&cpu_csr);
    let cpu_dense = cpu_csc.to_dense().expect("cpu dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    // GPU lane.
    let gpu_csr = CsrTensor::from_coo_on(&coo, Device::Cuda(0)).expect("gpu csr");
    let gpu_csc = CscTensor::from_csr_on(&gpu_csr, Device::Cuda(0)).expect("gpu csc");
    let gpu_dense = gpu_csc.to_dense_on(Device::Cuda(0)).expect("gpu dense");
    let gpu_data = gpu_dense.cpu().unwrap().data().unwrap().to_vec();

    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F32_TOL,
            "1024x1024 elem {i}: gpu={a} cpu={b}"
        );
    }
}

#[test]
fn p7_large_1024_density_1pct_cuda_f64() {
    ensure_cuda_backend();
    let m = 1024usize;
    let n = 1024usize;

    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for i in 0..m {
        for j in 0..n {
            if (i * 31 + j * 47) % 100 == 0 {
                rows.push(i);
                cols.push(j);
                vals.push(((i + 2 * j) as f64 * 0.0001) + 0.5);
            }
        }
    }
    let coo = CooTensor::<f64>::new(rows, cols, vals, m, n).expect("large coo");

    let cpu_csr = CsrTensor::from_coo(&coo).expect("cpu csr");
    let cpu_csc = CscTensor::from_csr(&cpu_csr);
    let cpu_dense = cpu_csc.to_dense().expect("cpu dense");
    let cpu_data = cpu_dense.data().expect("cpu data").to_vec();

    let gpu_csr = CsrTensor::from_coo_on(&coo, Device::Cuda(0)).expect("gpu csr");
    let gpu_csc = CscTensor::from_csr_on(&gpu_csr, Device::Cuda(0)).expect("gpu csc");
    let gpu_dense = gpu_csc.to_dense_on(Device::Cuda(0)).expect("gpu dense");
    let gpu_data = gpu_dense.cpu().unwrap().data().unwrap().to_vec();

    for (i, (&a, &b)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (a - b).abs() < F64_TOL,
            "1024x1024 f64 elem {i}: gpu={a} cpu={b}"
        );
    }
}
