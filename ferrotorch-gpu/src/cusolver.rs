//! cuSOLVER-backed GPU linear algebra: SVD, Cholesky, QR, Solve.
//!
//! Each operation follows the cuSOLVER pattern:
//! 1. Query workspace size via `*_bufferSize`.
//! 2. Allocate workspace + output buffers on the device.
//! 3. Call the cuSOLVER routine.
//! 4. Check `devInfo` — non-zero means the operation failed (singular matrix, etc.).
//!
//! All functions operate on column-major data because cuSOLVER (LAPACK-style)
//! uses column-major layout. The caller is responsible for transposing
//! row-major tensors before calling and transposing outputs back.

#[cfg(feature = "cuda")]
use cudarc::cusolver as cusolver_safe;
#[cfg(feature = "cuda")]
use cudarc::driver::{DevicePtr, DevicePtrMut};

#[cfg(feature = "cuda")]
use crate::buffer::CudaBuffer;
#[cfg(feature = "cuda")]
use crate::device::GpuDevice;
#[cfg(feature = "cuda")]
use crate::error::{GpuError, GpuResult};

// ---------------------------------------------------------------------------
// Helper: transpose row-major <-> column-major in-place on CPU
// ---------------------------------------------------------------------------

/// Transpose an m-by-n row-major flat array to column-major (or vice versa).
fn transpose_f32(data: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = data[i * n + j];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Helper: check devInfo (download single i32 from GPU, verify == 0)
// ---------------------------------------------------------------------------

/// Download a single i32 `devInfo` value from the GPU and check it.
///
/// Returns `Ok(info_val)` always. The caller decides how to interpret non-zero.
#[cfg(feature = "cuda")]
fn read_dev_info(info_buf: &CudaBuffer<i32>, device: &GpuDevice) -> GpuResult<i32> {
    let host = crate::transfer::gpu_to_cpu(info_buf, device)?;
    Ok(host[0])
}

/// Allocate a zero-initialized `CudaBuffer<i32>` (for devInfo / ipiv).
#[cfg(feature = "cuda")]
fn alloc_zeros_i32(len: usize, device: &GpuDevice) -> GpuResult<CudaBuffer<i32>> {
    crate::transfer::alloc_zeros::<i32>(len, device)
}

/// Allocate a `CudaBuffer<f32>` from host data.
#[cfg(feature = "cuda")]
fn upload_f32(data: &[f32], device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    crate::transfer::cpu_to_gpu(data, device)
}

/// Download a `CudaBuffer<f32>` to host.
#[cfg(feature = "cuda")]
fn download_f32(buf: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<Vec<f32>> {
    crate::transfer::gpu_to_cpu(buf, device)
}

// ---------------------------------------------------------------------------
// SVD: A = U * diag(S) * Vh   (thin/reduced)
// ---------------------------------------------------------------------------

/// Compute the thin SVD of an m-by-n matrix (row-major f32).
///
/// Returns `(U, S, Vh)` as flat row-major `Vec<f32>` with shapes:
/// - U:  [m, k]  where k = min(m, n)
/// - S:  [k]
/// - Vh: [k, n]
///
/// cuSOLVER's `Sgesvd` operates on column-major data and produces
/// column-major U and VT. We transpose on input and output.
#[cfg(feature = "cuda")]
pub fn gpu_svd_f32(
    data: &[f32],
    m: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    use cudarc::cusolver::sys as csys;

    if m == 0 || n == 0 {
        return Ok((vec![], vec![], vec![]));
    }

    let k = m.min(n);
    let stream = device.stream();

    // Create cuSOLVER handle.
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Transpose input from row-major to column-major.
    let col_major = transpose_f32(data, m, n);
    let mut d_a = upload_f32(&col_major, device)?;

    // Allocate output buffers on device.
    let mut d_s = crate::transfer::alloc_zeros_f32(k, device)?;
    let mut d_u = crate::transfer::alloc_zeros_f32(m * k, device)?;
    let mut d_vt = crate::transfer::alloc_zeros_f32(k * n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    // Query workspace size.
    let mut lwork: i32 = 0;
    // SAFETY: dn.cu() is a valid cusolverDnHandle_t, m/n are valid dimensions.
    unsafe {
        csys::cusolverDnSgesvd_bufferSize(
            dn.cu(),
            m as i32,
            n as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f32(lwork.max(1) as usize, device)?;

    // cuSOLVER Sgesvd: jobu='S' (thin U), jobvt='S' (thin VT).
    // SAFETY: All device pointers are valid allocations of the required sizes.
    // The handle and stream are valid. We synchronize and check devInfo after.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(stream);
        let (s_ptr, _s_sync) = d_s.inner_mut().device_ptr_mut(stream);
        let (u_ptr, _u_sync) = d_u.inner_mut().device_ptr_mut(stream);
        let (vt_ptr, _vt_sync) = d_vt.inner_mut().device_ptr_mut(stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(stream);

        csys::cusolverDnSgesvd(
            dn.cu(),
            b'S' as i8,   // jobu: thin U
            b'S' as i8,   // jobvt: thin VT
            m as i32,
            n as i32,
            a_ptr as *mut f32,
            m as i32,      // lda = m (column-major)
            s_ptr as *mut f32,
            u_ptr as *mut f32,
            m as i32,      // ldu = m
            vt_ptr as *mut f32,
            k as i32,      // ldvt = k (for thin SVD)
            work_ptr as *mut f32,
            lwork,
            std::ptr::null_mut(), // rwork (unused for real)
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    // Check devInfo.
    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_svd_f32",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    // Download results and transpose from column-major back to row-major.
    let s_host = download_f32(&d_s, device)?;

    // U is m-by-k column-major -> transpose to k columns, m rows row-major.
    let u_col = download_f32(&d_u, device)?;
    // Column-major m-by-k means the data is laid out as k columns of m elements.
    // To convert to row-major m-by-k: out[i*k + j] = col[j*m + i].
    let mut u_host = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            u_host[i * k + j] = u_col[j * m + i];
        }
    }

    // VT is k-by-n column-major -> convert to row-major k-by-n.
    let vt_col = download_f32(&d_vt, device)?;
    let mut vt_host = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            vt_host[i * n + j] = vt_col[j * k + i];
        }
    }

    Ok((u_host, s_host, vt_host))
}

// ---------------------------------------------------------------------------
// Cholesky: A = L * L^T   (lower-triangular)
// ---------------------------------------------------------------------------

/// Compute the Cholesky decomposition of an n-by-n SPD matrix (row-major f32).
///
/// Returns the lower-triangular factor L as a flat row-major `Vec<f32>` [n, n].
///
/// Upper-triangular entries are explicitly zeroed.
#[cfg(feature = "cuda")]
pub fn gpu_cholesky_f32(
    data: &[f32],
    n: usize,
    device: &GpuDevice,
) -> GpuResult<Vec<f32>> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        return Ok(vec![]);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Transpose to column-major.
    let col_major = transpose_f32(data, n, n);
    let mut d_a = upload_f32(&col_major, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    // Query workspace size.
    let mut lwork: i32 = 0;
    // SAFETY: dn.cu() is valid, d_a points to n*n f32 elements.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(stream);
        csys::cusolverDnSpotrf_bufferSize(
            dn.cu(),
            csys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
            n as i32,
            a_ptr as *mut f32,
            n as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f32(lwork.max(1) as usize, device)?;

    // SAFETY: All device pointers are valid. We use LOWER fill mode.
    // devInfo is checked after synchronization.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(stream);

        csys::cusolverDnSpotrf(
            dn.cu(),
            csys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
            n as i32,
            a_ptr as *mut f32,
            n as i32,
            work_ptr as *mut f32,
            lwork,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_cholesky_f32: matrix is not positive-definite",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    // Download column-major result and convert to row-major.
    let l_col = download_f32(&d_a, device)?;
    let mut l_host = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            l_host[i * n + j] = l_col[j * n + i];
        }
    }

    // cuSOLVER only writes the lower triangle; zero the upper triangle explicitly.
    for i in 0..n {
        for j in (i + 1)..n {
            l_host[i * n + j] = 0.0;
        }
    }

    Ok(l_host)
}

// ---------------------------------------------------------------------------
// Solve: A * X = B   (via LU factorization: getrf + getrs)
// ---------------------------------------------------------------------------

/// Solve A * X = B for X where A is n-by-n and B is n-by-nrhs (row-major f32).
///
/// Uses LU factorization (Sgetrf) followed by triangular solve (Sgetrs).
///
/// Returns X as flat row-major `Vec<f32>` with shape [n, nrhs] (or [n] if nrhs==1).
#[cfg(feature = "cuda")]
pub fn gpu_solve_f32(
    a_data: &[f32],
    b_data: &[f32],
    n: usize,
    nrhs: usize,
    device: &GpuDevice,
) -> GpuResult<Vec<f32>> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        return Ok(vec![]);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Convert A to column-major.
    let a_col = transpose_f32(a_data, n, n);
    let mut d_a = upload_f32(&a_col, device)?;

    // Convert B to column-major (n-by-nrhs).
    let b_col = transpose_f32(b_data, n, nrhs);
    let mut d_b = upload_f32(&b_col, device)?;

    let mut d_ipiv = alloc_zeros_i32(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    // Query workspace for getrf.
    let mut lwork: i32 = 0;
    // SAFETY: dn.cu() is valid, d_a contains n*n f32 elements.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(stream);
        csys::cusolverDnSgetrf_bufferSize(
            dn.cu(),
            n as i32,
            n as i32,
            a_ptr as *mut f32,
            n as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f32(lwork.max(1) as usize, device)?;

    // LU factorization: A = P * L * U.
    // SAFETY: All device pointers are valid allocations of the required sizes.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner_mut().device_ptr_mut(stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(stream);

        csys::cusolverDnSgetrf(
            dn.cu(),
            n as i32,
            n as i32,
            a_ptr as *mut f32,
            n as i32,
            work_ptr as *mut f32,
            ipiv_ptr as *mut i32,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f32: LU factorization failed (singular matrix)",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    // Triangular solve: L * U * X = P * B.
    // Reset devInfo for getrs.
    let mut d_info2 = alloc_zeros_i32(1, device)?;

    // SAFETY: d_a now contains the LU factors, d_ipiv the pivot indices,
    // d_b will be overwritten with the solution X. All are properly sized.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner().device_ptr(stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner().device_ptr(stream);
        let (b_ptr, _b_sync) = d_b.inner_mut().device_ptr_mut(stream);
        let (info_ptr, _info_sync) = d_info2.inner_mut().device_ptr_mut(stream);

        csys::cusolverDnSgetrs(
            dn.cu(),
            csys::cublasOperation_t::CUBLAS_OP_N, // no transpose
            n as i32,
            nrhs as i32,
            a_ptr as *const f32,
            n as i32,
            ipiv_ptr as *const i32,
            b_ptr as *mut f32,
            n as i32,  // ldb = n
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val2 = read_dev_info(&d_info2, device)?;
    if info_val2 != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f32: triangular solve failed",
            expected: vec![0],
            got: vec![info_val2 as usize],
        });
    }

    // Download solution (column-major) and convert to row-major.
    let x_col = download_f32(&d_b, device)?;
    let mut x_host = vec![0.0f32; n * nrhs];
    for i in 0..n {
        for j in 0..nrhs {
            x_host[i * nrhs + j] = x_col[j * n + i];
        }
    }

    Ok(x_host)
}

// ---------------------------------------------------------------------------
// QR: A = Q * R   (reduced/thin)
// ---------------------------------------------------------------------------

/// Compute the reduced QR decomposition of an m-by-n matrix (row-major f32).
///
/// Returns `(Q, R)` as flat row-major `Vec<f32>` with shapes:
/// - Q: [m, k]  where k = min(m, n)
/// - R: [k, n]
///
/// Uses Sgeqrf (Householder QR) followed by Sorgqr (generate Q).
#[cfg(feature = "cuda")]
pub fn gpu_qr_f32(
    data: &[f32],
    m: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(Vec<f32>, Vec<f32>)> {
    use cudarc::cusolver::sys as csys;

    if m == 0 || n == 0 {
        return Ok((vec![], vec![]));
    }

    let k = m.min(n);
    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Transpose to column-major.
    let col_major = transpose_f32(data, m, n);
    let mut d_a = upload_f32(&col_major, device)?;
    let mut d_tau = crate::transfer::alloc_zeros_f32(k, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    // Query workspace for geqrf.
    let mut lwork: i32 = 0;
    // SAFETY: dn.cu() is valid, d_a contains m*n f32 elements.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(stream);
        csys::cusolverDnSgeqrf_bufferSize(
            dn.cu(),
            m as i32,
            n as i32,
            a_ptr as *mut f32,
            m as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f32(lwork.max(1) as usize, device)?;

    // Compute QR factorization (Householder form).
    // SAFETY: All device pointers are valid. d_a is overwritten in-place
    // with Householder reflectors (lower triangle) and R (upper triangle).
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(stream);
        let (tau_ptr, _tau_sync) = d_tau.inner_mut().device_ptr_mut(stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(stream);

        csys::cusolverDnSgeqrf(
            dn.cu(),
            m as i32,
            n as i32,
            a_ptr as *mut f32,
            m as i32,
            tau_ptr as *mut f32,
            work_ptr as *mut f32,
            lwork,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_qr_f32: geqrf failed",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    // Extract R from the upper triangle of d_a (column-major).
    // R is k-by-n. We read the full m-by-n column-major buffer.
    let qr_col = download_f32(&d_a, device)?;
    let mut r_host = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            // In column-major m-by-n: element (i, j) is at index j*m + i.
            if j >= i {
                r_host[i * n + j] = qr_col[j * m + i]; // row-major output
            }
            // else: R[i,j] = 0 (already initialized)
        }
    }

    // Generate explicit Q via Sorgqr.
    // Sorgqr overwrites d_a in-place: the first k columns become Q (m-by-k, column-major).
    let mut lwork_orgqr: i32 = 0;

    // SAFETY: dn.cu() is valid, d_a and d_tau contain valid QR factorization data.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner().device_ptr(stream);
        let (tau_ptr, _tau_sync) = d_tau.inner().device_ptr(stream);
        csys::cusolverDnSorgqr_bufferSize(
            dn.cu(),
            m as i32,
            k as i32,
            k as i32,
            a_ptr as *const f32,
            m as i32,
            tau_ptr as *const f32,
            &mut lwork_orgqr,
        )
        .result()?;
    }

    let mut d_work2 = crate::transfer::alloc_zeros_f32(lwork_orgqr.max(1) as usize, device)?;
    let mut d_info2 = alloc_zeros_i32(1, device)?;

    // SAFETY: d_a contains the Householder reflectors from geqrf, d_tau the
    // scalar factors. Sorgqr overwrites the first k columns of d_a with Q.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(stream);
        let (tau_ptr, _tau_sync) = d_tau.inner().device_ptr(stream);
        let (work_ptr, _work_sync) = d_work2.inner_mut().device_ptr_mut(stream);
        let (info_ptr, _info_sync) = d_info2.inner_mut().device_ptr_mut(stream);

        csys::cusolverDnSorgqr(
            dn.cu(),
            m as i32,
            k as i32,
            k as i32,
            a_ptr as *mut f32,
            m as i32,
            tau_ptr as *const f32,
            work_ptr as *mut f32,
            lwork_orgqr,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val2 = read_dev_info(&d_info2, device)?;
    if info_val2 != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_qr_f32: orgqr failed",
            expected: vec![0],
            got: vec![info_val2 as usize],
        });
    }

    // Download Q (m-by-k column-major from d_a, but d_a has n columns total;
    // we only need the first k columns).
    let q_full_col = download_f32(&d_a, device)?;
    let mut q_host = vec![0.0f32; m * k];
    for i in 0..m {
        for j in 0..k {
            q_host[i * k + j] = q_full_col[j * m + i]; // col-major -> row-major
        }
    }

    Ok((q_host, r_host))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    fn device() -> GpuDevice {
        GpuDevice::new(0).expect("CUDA device 0")
    }

    // -- SVD tests --

    #[test]
    fn svd_reconstructs_3x2() {
        let dev = device();
        // A = [[1, 2], [3, 4], [5, 6]]
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (m, n) = (3, 2);
        let (u, s, vt) = gpu_svd_f32(&a, m, n, &dev).unwrap();
        let k = m.min(n);

        assert_eq!(u.len(), m * k);
        assert_eq!(s.len(), k);
        assert_eq!(vt.len(), k * n);

        // Reconstruct: U * diag(S) * VT
        let mut recon = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += u[i * k + p] * s[p] * vt[p * n + j];
                }
                recon[i * n + j] = acc;
            }
        }

        for i in 0..m * n {
            assert!(
                (recon[i] - a[i]).abs() < 1e-3,
                "SVD reconstruction failed at {i}: {} vs {}",
                recon[i],
                a[i]
            );
        }
    }

    #[test]
    fn svd_singular_values_descending() {
        let dev = device();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (_, s, _) = gpu_svd_f32(&a, 3, 2, &dev).unwrap();
        assert!(s[0] >= s[1], "singular values must be descending");
    }

    #[test]
    fn svd_square_identity() {
        let dev = device();
        let eye: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (_, s, _) = gpu_svd_f32(&eye, 3, 3, &dev).unwrap();
        for &sv in &s {
            assert!((sv - 1.0).abs() < 1e-4, "identity SVD should have all ones");
        }
    }

    #[test]
    fn svd_empty() {
        let dev = device();
        let (u, s, vt) = gpu_svd_f32(&[], 0, 0, &dev).unwrap();
        assert!(u.is_empty());
        assert!(s.is_empty());
        assert!(vt.is_empty());
    }

    // -- Cholesky tests --

    #[test]
    fn cholesky_spd_3x3() {
        let dev = device();
        // SPD matrix: A = [[6,5,1],[5,12,5],[1,5,6]]
        #[rustfmt::skip]
        let a: Vec<f32> = vec![
            6.0, 5.0, 1.0,
            5.0, 12.0, 5.0,
            1.0, 5.0, 6.0,
        ];
        let l = gpu_cholesky_f32(&a, 3, &dev).unwrap();

        // Verify lower-triangular.
        for i in 0..3 {
            for j in (i + 1)..3 {
                assert!(
                    l[i * 3 + j].abs() < 1e-5,
                    "L[{i},{j}] = {} should be 0",
                    l[i * 3 + j]
                );
            }
        }

        // Reconstruct: L * L^T should equal A.
        let mut llt = vec![0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0f32;
                for p in 0..3 {
                    acc += l[i * 3 + p] * l[j * 3 + p];
                }
                llt[i * 3 + j] = acc;
            }
        }

        for i in 0..9 {
            assert!(
                (llt[i] - a[i]).abs() < 1e-3,
                "L*L^T[{i}] = {} vs A[{i}] = {}",
                llt[i],
                a[i]
            );
        }
    }

    #[test]
    fn cholesky_identity() {
        let dev = device();
        let eye: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let l = gpu_cholesky_f32(&eye, 3, &dev).unwrap();
        // Cholesky of identity is identity.
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (l[i * 3 + j] - expected).abs() < 1e-5,
                    "L[{i},{j}] = {} (expected {})",
                    l[i * 3 + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn cholesky_empty() {
        let dev = device();
        let l = gpu_cholesky_f32(&[], 0, &dev).unwrap();
        assert!(l.is_empty());
    }

    // -- Solve tests --

    #[test]
    fn solve_2x2_simple() {
        let dev = device();
        // A = [[2, 1], [1, 3]], b = [5, 10]
        // Solution: x = [1, 3]
        let a: Vec<f32> = vec![2.0, 1.0, 1.0, 3.0];
        let b: Vec<f32> = vec![5.0, 10.0];
        let x = gpu_solve_f32(&a, &b, 2, 1, &dev).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-3, "x[0] = {} (expected 1.0)", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-3, "x[1] = {} (expected 3.0)", x[1]);
    }

    #[test]
    fn solve_identity() {
        let dev = device();
        let eye: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let b: Vec<f32> = vec![7.0, 11.0];
        let x = gpu_solve_f32(&eye, &b, 2, 1, &dev).unwrap();
        assert!((x[0] - 7.0).abs() < 1e-4);
        assert!((x[1] - 11.0).abs() < 1e-4);
    }

    #[test]
    fn solve_multiple_rhs() {
        let dev = device();
        // A = [[2, 1], [1, 3]], B = [[5, 3], [10, 7]]
        // X = A^-1 * B
        let a: Vec<f32> = vec![2.0, 1.0, 1.0, 3.0];
        let b: Vec<f32> = vec![5.0, 3.0, 10.0, 7.0]; // 2x2 row-major
        let x = gpu_solve_f32(&a, &b, 2, 2, &dev).unwrap();
        // Verify: A * X should equal B
        let mut ax = vec![0.0f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                ax[i * 2 + j] = a[i * 2] * x[j] + a[i * 2 + 1] * x[2 + j];
            }
        }
        for i in 0..4 {
            assert!(
                (ax[i] - b[i]).abs() < 1e-3,
                "A*X[{i}] = {} vs B[{i}] = {}",
                ax[i],
                b[i]
            );
        }
    }

    #[test]
    fn solve_empty() {
        let dev = device();
        let x = gpu_solve_f32(&[], &[], 0, 0, &dev).unwrap();
        assert!(x.is_empty());
    }

    // -- QR tests --

    #[test]
    fn qr_reconstructs_3x2() {
        let dev = device();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (m, n) = (3, 2);
        let (q, r) = gpu_qr_f32(&a, m, n, &dev).unwrap();
        let k = m.min(n);

        assert_eq!(q.len(), m * k);
        assert_eq!(r.len(), k * n);

        // Reconstruct: Q * R should equal A.
        let mut recon = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += q[i * k + p] * r[p * n + j];
                }
                recon[i * n + j] = acc;
            }
        }

        for i in 0..m * n {
            assert!(
                (recon[i] - a[i]).abs() < 1e-3,
                "QR reconstruction failed at {i}: {} vs {}",
                recon[i],
                a[i]
            );
        }
    }

    #[test]
    fn qr_orthogonal_q() {
        let dev = device();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (m, n) = (3, 2);
        let (q, _) = gpu_qr_f32(&a, m, n, &dev).unwrap();
        let k = m.min(n);

        // Q^T * Q should be identity_k.
        let mut qtq = vec![0.0f32; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0f32;
                for p in 0..m {
                    acc += q[p * k + i] * q[p * k + j];
                }
                qtq[i * k + j] = acc;
            }
        }

        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[i * k + j] - expected).abs() < 1e-3,
                    "Q^T*Q[{i},{j}] = {} (expected {})",
                    qtq[i * k + j],
                    expected
                );
            }
        }
    }

    #[test]
    fn qr_r_upper_triangular() {
        let dev = device();
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (_, r) = gpu_qr_f32(&a, 3, 2, &dev).unwrap();
        let k = 2;
        let n = 2;
        // R should be upper-triangular.
        for i in 0..k {
            for j in 0..i.min(n) {
                assert!(
                    r[i * n + j].abs() < 1e-4,
                    "R[{i},{j}] = {} should be 0",
                    r[i * n + j]
                );
            }
        }
    }

    #[test]
    fn qr_square() {
        let dev = device();
        // 3x3 matrix
        let a: Vec<f32> = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let (q, r) = gpu_qr_f32(&a, 3, 3, &dev).unwrap();
        let k = 3;
        let n = 3;

        // Reconstruct
        let mut recon = vec![0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += q[i * k + p] * r[p * n + j];
                }
                recon[i * n + j] = acc;
            }
        }
        for i in 0..9 {
            assert!(
                (recon[i] - a[i]).abs() < 1e-3,
                "QR square reconstruction failed at {i}: {} vs {}",
                recon[i],
                a[i]
            );
        }
    }

    #[test]
    fn qr_empty() {
        let dev = device();
        let (q, r) = gpu_qr_f32(&[], 0, 0, &dev).unwrap();
        assert!(q.is_empty());
        assert!(r.is_empty());
    }
}
