//! GPU `lu_factor` integration tests against the RTX 3090. (#604)
//!
//! Validates the cuSOLVER-backed `gpu_lu_factor_f32` / `gpu_lu_factor_f64`
//! against the CPU reference (`ferray-linalg::lu`) and against the closed
//! contract that `P @ L @ U == A` reconstructs the original matrix.

#![cfg(feature = "cuda")]

use ferrotorch_core::linalg;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn cpu_t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

/// Reconstruct A from packed LU + pivots. Used to validate the GPU path
/// without comparing element-by-element against a non-deterministic CPU
/// reference (different LAPACK / ferray implementations may pick equivalent
/// but distinct factorizations).
fn reconstruct_from_lu_factor_f32(lu: &[f32], ipiv: &[i32], n: usize) -> Vec<f32> {
    // L: strict lower triangle of LU + unit diagonal
    let mut l = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                l[i * n + j] = 1.0;
            } else if j < i {
                l[i * n + j] = lu[i * n + j];
            }
        }
    }
    // U: upper triangle (incl. diagonal) of LU
    let mut u = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = lu[i * n + j];
        }
    }
    // L @ U
    let mut lu_prod = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += l[i * n + k] * u[k * n + j];
            }
            lu_prod[i * n + j] = acc;
        }
    }
    // Apply inverse permutation P^T (i.e. swap rows in the order ipiv tells us).
    // ipiv[i] is the 1-based row that was swapped INTO row i during
    // factorization. To get A = P^{-1} (LU) we apply the swaps in reverse.
    let mut a = lu_prod;
    for i in (0..n).rev() {
        let p = (ipiv[i] - 1) as usize;
        if p != i {
            for j in 0..n {
                a.swap(i * n + j, p * n + j);
            }
        }
    }
    a
}

fn reconstruct_from_lu_factor_f64(lu: &[f64], ipiv: &[i32], n: usize) -> Vec<f64> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                l[i * n + j] = 1.0;
            } else if j < i {
                l[i * n + j] = lu[i * n + j];
            }
        }
    }
    let mut u = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = lu[i * n + j];
        }
    }
    let mut lu_prod = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += l[i * n + k] * u[k * n + j];
            }
            lu_prod[i * n + j] = acc;
        }
    }
    let mut a = lu_prod;
    for i in (0..n).rev() {
        let p = (ipiv[i] - 1) as usize;
        if p != i {
            for j in 0..n {
                a.swap(i * n + j, p * n + j);
            }
        }
    }
    a
}

#[test]
fn lu_factor_f32_3x3_reconstructs_input() {
    ensure_cuda();
    // A non-singular 3x3 matrix with mixed magnitudes.
    let a_data: Vec<f32> = vec![
        2.0, 1.0, 1.0, // row 0
        4.0, 3.0, 3.0, // row 1
        8.0, 7.0, 9.0, // row 2
    ];
    let a_cpu = cpu_t_f32(&a_data, &[3, 3]);
    let a_gpu = a_cpu.to(Device::Cuda(0)).unwrap();

    let (lu, ipiv) = linalg::lu_factor(&a_gpu).unwrap();
    assert_eq!(lu.shape(), &[3, 3]);
    assert_eq!(ipiv.len(), 3);
    // LU stays on GPU.
    assert!(matches!(lu.device(), Device::Cuda(0)));

    // Reconstruct A from the factorization and compare element-wise.
    let lu_host = lu.cpu().unwrap().data().unwrap().to_vec();
    let recon = reconstruct_from_lu_factor_f32(&lu_host, &ipiv, 3);
    for (i, (got, expected)) in recon.iter().zip(a_data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "mismatch at {i}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn lu_factor_f64_5x5_reconstructs_input() {
    ensure_cuda();
    // A 5x5 well-conditioned matrix.
    let a_data: Vec<f64> = vec![
        4.0, 3.0, 0.0, 0.0, 1.0, //
        6.0, 3.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 5.0, 2.0, 0.0, //
        0.0, 0.0, 2.0, 4.0, 0.0, //
        1.0, 0.0, 0.0, 1.0, 7.0, //
    ];
    let a_cpu = cpu_t_f64(&a_data, &[5, 5]);
    let a_gpu = a_cpu.to(Device::Cuda(0)).unwrap();

    let (lu, ipiv) = linalg::lu_factor(&a_gpu).unwrap();
    assert_eq!(lu.shape(), &[5, 5]);
    assert_eq!(ipiv.len(), 5);

    let lu_host = lu.cpu().unwrap().data().unwrap().to_vec();
    let recon = reconstruct_from_lu_factor_f64(&lu_host, &ipiv, 5);
    for (i, (got, expected)) in recon.iter().zip(a_data.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-9,
            "f64 mismatch at {i}: got {got}, expected {expected}"
        );
    }
}

#[test]
fn lu_factor_cpu_path_matches_gpu_via_reconstruction() {
    ensure_cuda();
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]; // non-singular
    let a_cpu = cpu_t_f32(&a_data, &[3, 3]);

    let (cpu_lu, cpu_ipiv) = linalg::lu_factor(&a_cpu).unwrap();
    let cpu_recon = reconstruct_from_lu_factor_f32(cpu_lu.data().unwrap(), &cpu_ipiv, 3);

    let a_gpu = a_cpu.to(Device::Cuda(0)).unwrap();
    let (gpu_lu, gpu_ipiv) = linalg::lu_factor(&a_gpu).unwrap();
    let gpu_lu_host = gpu_lu.cpu().unwrap().data().unwrap().to_vec();
    let gpu_recon = reconstruct_from_lu_factor_f32(&gpu_lu_host, &gpu_ipiv, 3);

    // Both reconstruct A even though the factorizations may differ in
    // pivot order — element-wise equality on the reconstructions is the
    // honest correctness check.
    for (i, (g, c)) in gpu_recon.iter().zip(cpu_recon.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-3,
            "recon mismatch at {i}: gpu={g} cpu={c}"
        );
    }
}

#[test]
fn lu_factor_rejects_non_square() {
    ensure_cuda();
    let a = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
        .to(Device::Cuda(0))
        .unwrap();
    let err = linalg::lu_factor(&a).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("square"), "got: {msg}");
}

#[test]
fn lu_factor_large_matrix_64x64_reconstructs() {
    ensure_cuda();
    // Build a strongly diagonal 64x64 matrix so it's non-singular.
    let n = 64;
    let mut a_data = vec![0.0_f32; n * n];
    for i in 0..n {
        a_data[i * n + i] = (i + 1) as f32 * 2.0;
        // Add some off-diagonal noise so the LU isn't trivial.
        if i + 1 < n {
            a_data[i * n + (i + 1)] = 0.5;
        }
        if i > 0 {
            a_data[i * n + (i - 1)] = 0.3;
        }
    }
    let a_cpu = cpu_t_f32(&a_data, &[n, n]);
    let a_gpu = a_cpu.to(Device::Cuda(0)).unwrap();

    let (lu, ipiv) = linalg::lu_factor(&a_gpu).unwrap();
    assert_eq!(lu.shape(), &[n, n]);
    assert_eq!(ipiv.len(), n);

    let lu_host = lu.cpu().unwrap().data().unwrap().to_vec();
    let recon = reconstruct_from_lu_factor_f32(&lu_host, &ipiv, n);
    let mut max_err = 0.0_f32;
    for (g, c) in recon.iter().zip(a_data.iter()) {
        max_err = max_err.max((g - c).abs());
    }
    assert!(
        max_err < 1e-3,
        "max recon error {max_err} too large for 64x64"
    );
}
