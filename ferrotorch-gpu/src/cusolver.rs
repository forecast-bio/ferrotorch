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

#[cfg(not(feature = "cuda"))]
use crate::device::GpuDevice;
#[cfg(not(feature = "cuda"))]
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

/// Transpose an m-by-n row-major flat array to column-major (or vice versa) — f64 variant.
fn transpose_f64(data: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
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

/// Allocate a `CudaBuffer<f64>` from host data.
#[cfg(feature = "cuda")]
fn upload_f64(data: &[f64], device: &GpuDevice) -> GpuResult<CudaBuffer<f64>> {
    crate::transfer::cpu_to_gpu(data, device)
}

/// Download a `CudaBuffer<f64>` to host.
#[cfg(feature = "cuda")]
fn download_f64(buf: &CudaBuffer<f64>, device: &GpuDevice) -> GpuResult<Vec<f64>> {
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
        csys::cusolverDnSgesvd_bufferSize(dn.cu(), m as i32, n as i32, &mut lwork).result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f32(lwork.max(1) as usize, device)?;

    // cuSOLVER Sgesvd: jobu='S' (thin U), jobvt='S' (thin VT).
    // SAFETY: All device pointers are valid allocations of the required sizes.
    // The handle and stream are valid. We synchronize and check devInfo after.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (s_ptr, _s_sync) = d_s.inner_mut().device_ptr_mut(&stream);
        let (u_ptr, _u_sync) = d_u.inner_mut().device_ptr_mut(&stream);
        let (vt_ptr, _vt_sync) = d_vt.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnSgesvd(
            dn.cu(),
            b'S' as i8, // jobu: thin U
            b'S' as i8, // jobvt: thin VT
            m as i32,
            n as i32,
            a_ptr as *mut f32,
            m as i32, // lda = m (column-major)
            s_ptr as *mut f32,
            u_ptr as *mut f32,
            m as i32, // ldu = m
            vt_ptr as *mut f32,
            k as i32, // ldvt = k (for thin SVD)
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

/// Compute the thin SVD of an m-by-n matrix (row-major f64).
///
/// Returns `(U, S, Vh)` as flat row-major `Vec<f64>` with shapes:
/// - U:  [m, k]  where k = min(m, n)
/// - S:  [k]
/// - Vh: [k, n]
///
/// cuSOLVER's `Dgesvd` operates on column-major data and produces
/// column-major U and VT. We transpose on input and output.
#[cfg(feature = "cuda")]
pub fn gpu_svd_f64(
    data: &[f64],
    m: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    use cudarc::cusolver::sys as csys;

    if m == 0 || n == 0 {
        return Ok((vec![], vec![], vec![]));
    }

    let k = m.min(n);
    let stream = device.stream();

    // Create cuSOLVER handle.
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Transpose input from row-major to column-major.
    let col_major = transpose_f64(data, m, n);
    let mut d_a = upload_f64(&col_major, device)?;

    // Allocate output buffers on device.
    let mut d_s = crate::transfer::alloc_zeros_f64(k, device)?;
    let mut d_u = crate::transfer::alloc_zeros_f64(m * k, device)?;
    let mut d_vt = crate::transfer::alloc_zeros_f64(k * n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    // Query workspace size.
    let mut lwork: i32 = 0;
    // SAFETY: dn.cu() is a valid cusolverDnHandle_t, m/n are valid dimensions.
    unsafe {
        csys::cusolverDnDgesvd_bufferSize(dn.cu(), m as i32, n as i32, &mut lwork).result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f64(lwork.max(1) as usize, device)?;

    // cuSOLVER Dgesvd: jobu='S' (thin U), jobvt='S' (thin VT).
    // SAFETY: All device pointers are valid allocations of the required sizes.
    // The handle and stream are valid. We synchronize and check devInfo after.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (s_ptr, _s_sync) = d_s.inner_mut().device_ptr_mut(&stream);
        let (u_ptr, _u_sync) = d_u.inner_mut().device_ptr_mut(&stream);
        let (vt_ptr, _vt_sync) = d_vt.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDgesvd(
            dn.cu(),
            b'S' as i8, // jobu: thin U
            b'S' as i8, // jobvt: thin VT
            m as i32,
            n as i32,
            a_ptr as *mut f64,
            m as i32, // lda = m (column-major)
            s_ptr as *mut f64,
            u_ptr as *mut f64,
            m as i32, // ldu = m
            vt_ptr as *mut f64,
            k as i32, // ldvt = k (for thin SVD)
            work_ptr as *mut f64,
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
            op: "gpu_svd_f64",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    // Download results and transpose from column-major back to row-major.
    let s_host = download_f64(&d_s, device)?;

    // U is m-by-k column-major -> transpose to k columns, m rows row-major.
    let u_col = download_f64(&d_u, device)?;
    // Column-major m-by-k means the data is laid out as k columns of m elements.
    // To convert to row-major m-by-k: out[i*k + j] = col[j*m + i].
    let mut u_host = vec![0.0f64; m * k];
    for i in 0..m {
        for j in 0..k {
            u_host[i * k + j] = u_col[j * m + i];
        }
    }

    // VT is k-by-n column-major -> convert to row-major k-by-n.
    let vt_col = download_f64(&d_vt, device)?;
    let mut vt_host = vec![0.0f64; k * n];
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
pub fn gpu_cholesky_f32(data: &[f32], n: usize, device: &GpuDevice) -> GpuResult<Vec<f32>> {
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
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
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
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

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

/// Compute the Cholesky decomposition of an n-by-n SPD matrix (row-major f64).
///
/// Returns the lower-triangular factor L as a flat row-major `Vec<f64>` [n, n].
///
/// Upper-triangular entries are explicitly zeroed.
#[cfg(feature = "cuda")]
pub fn gpu_cholesky_f64(data: &[f64], n: usize, device: &GpuDevice) -> GpuResult<Vec<f64>> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        return Ok(vec![]);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Transpose to column-major.
    let col_major = transpose_f64(data, n, n);
    let mut d_a = upload_f64(&col_major, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    // Query workspace size.
    let mut lwork: i32 = 0;
    // SAFETY: dn.cu() is valid, d_a points to n*n f64 elements.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnDpotrf_bufferSize(
            dn.cu(),
            csys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f64(lwork.max(1) as usize, device)?;

    // SAFETY: All device pointers are valid. We use LOWER fill mode.
    // devInfo is checked after synchronization.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDpotrf(
            dn.cu(),
            csys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            work_ptr as *mut f64,
            lwork,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_cholesky_f64: matrix is not positive-definite",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    // Download column-major result and convert to row-major.
    let l_col = download_f64(&d_a, device)?;
    let mut l_host = vec![0.0f64; n * n];
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
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
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
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

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
        let (a_ptr, _a_sync) = d_a.inner().device_ptr(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner().device_ptr(&stream);
        let (b_ptr, _b_sync) = d_b.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info2.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnSgetrs(
            dn.cu(),
            csys::cublasOperation_t::CUBLAS_OP_N, // no transpose
            n as i32,
            nrhs as i32,
            a_ptr as *const f32,
            n as i32,
            ipiv_ptr as *const i32,
            b_ptr as *mut f32,
            n as i32, // ldb = n
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

/// Solve A * X = B for X where A is n-by-n and B is n-by-nrhs (row-major f64).
///
/// Uses LU factorization (Dgetrf) followed by triangular solve (Dgetrs).
///
/// Returns X as flat row-major `Vec<f64>` with shape [n, nrhs] (or [n] if nrhs==1).
#[cfg(feature = "cuda")]
pub fn gpu_solve_f64(
    a_data: &[f64],
    b_data: &[f64],
    n: usize,
    nrhs: usize,
    device: &GpuDevice,
) -> GpuResult<Vec<f64>> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        return Ok(vec![]);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Convert A to column-major.
    let a_col = transpose_f64(a_data, n, n);
    let mut d_a = upload_f64(&a_col, device)?;

    // Convert B to column-major (n-by-nrhs).
    let b_col = transpose_f64(b_data, n, nrhs);
    let mut d_b = upload_f64(&b_col, device)?;

    let mut d_ipiv = alloc_zeros_i32(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    // Query workspace for getrf.
    let mut lwork: i32 = 0;
    // SAFETY: dn.cu() is valid, d_a contains n*n f64 elements.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnDgetrf_bufferSize(
            dn.cu(),
            n as i32,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f64(lwork.max(1) as usize, device)?;

    // LU factorization: A = P * L * U.
    // SAFETY: All device pointers are valid allocations of the required sizes.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDgetrf(
            dn.cu(),
            n as i32,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            work_ptr as *mut f64,
            ipiv_ptr as *mut i32,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f64: LU factorization failed (singular matrix)",
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
        let (a_ptr, _a_sync) = d_a.inner().device_ptr(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner().device_ptr(&stream);
        let (b_ptr, _b_sync) = d_b.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info2.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDgetrs(
            dn.cu(),
            csys::cublasOperation_t::CUBLAS_OP_N, // no transpose
            n as i32,
            nrhs as i32,
            a_ptr as *const f64,
            n as i32,
            ipiv_ptr as *const i32,
            b_ptr as *mut f64,
            n as i32, // ldb = n
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val2 = read_dev_info(&d_info2, device)?;
    if info_val2 != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f64: triangular solve failed",
            expected: vec![0],
            got: vec![info_val2 as usize],
        });
    }

    // Download solution (column-major) and convert to row-major.
    let x_col = download_f64(&d_b, device)?;
    let mut x_host = vec![0.0f64; n * nrhs];
    for i in 0..n {
        for j in 0..nrhs {
            x_host[i * nrhs + j] = x_col[j * n + i];
        }
    }

    Ok(x_host)
}

// ---------------------------------------------------------------------------
// LU factorization (packed): A = P * L * U  (#604)
// ---------------------------------------------------------------------------
//
// Returns the cuSOLVER native packed form:
// - LU buffer: an n×n row-major tensor where the strict lower-triangle is L
//   (unit diagonal implicit) and the upper-triangle (incl. diagonal) is U.
// - pivots: an i32 buffer of length n; 1-based row-permutation indices.
//
// Both outputs stay on device — no host bounce. Input is taken as
// `&CudaBuffer<f32/f64>` (row-major) and we use `gpu_transpose_2d` for the
// row→column→row roundtrip cuSOLVER demands.

/// GPU-resident LU factorization of an n×n f32 matrix (#604).
/// Mirrors `torch.linalg.lu_factor` for CUDA inputs.
#[cfg(feature = "cuda")]
pub fn gpu_lu_factor_f32(
    a_dev: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<i32>)> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        return Ok((
            crate::transfer::alloc_zeros_f32(0, device)?,
            alloc_zeros_i32(0, device)?,
        ));
    }
    if a_dev.len() != n * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_lu_factor_f32: input length mismatch",
            expected: vec![n * n],
            got: vec![a_dev.len()],
        });
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Row-major → column-major on device. This stays in VRAM end-to-end.
    let mut d_a_col = crate::kernels::gpu_transpose_2d(a_dev, n, n, device)?;
    let mut d_ipiv = alloc_zeros_i32(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 827 by `cusolverDnHandle::new(stream.clone())`. The
    //   handle's lifetime is tied to `dn`, which lives for the duration of
    //   this function — the FFI call completes before `dn` is dropped.
    // - `n` was validated at line 818-824 (early-return on `a_dev.len() != n*n`)
    //   and is non-zero (line 812 short-circuits the n==0 path).
    // - `a_ptr` points to an `n*n` f32 column-major buffer allocated at line 830
    //   by `gpu_transpose_2d`; `lda = n as i32` matches the column-major leading
    //   dimension as required by cuSOLVER's LAPACK-style ABI (lda >= max(1, n)).
    // - `&mut lwork` is a stack-resident `i32`; cuSOLVER writes the workspace
    //   size in elements (not bytes) per the upstream docs for `Sgetrf_bufferSize`.
    // - This is a query-only call: it inspects `m`, `n`, `lda` only and does not
    //   touch the matrix data, so the `device_ptr_mut` borrow is harmless.
    // - Column-major layout is upheld because `d_a_col` came from
    //   `gpu_transpose_2d(a_dev, n, n, ..)` at line 830 (row→col conversion).
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
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

    // SAFETY:
    // - `dn.cu()` is the same valid handle proved above.
    // - `a_ptr` points to the `n*n` column-major f32 matrix (line 830); cuSOLVER
    //   overwrites it in place with the LU factors (L below diag, U on/above).
    // - `lda = n as i32` matches the column-major leading dimension.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f32 elements
    //   allocated at line 848 from the `Sgetrf_bufferSize` query above; the
    //   `.max(1)` guards against the rare lwork==0 case so the alloc is never
    //   zero-sized. cuSOLVER receives `lwork` (the queried value) so the
    //   buffer is at least the size cuSOLVER claimed it needs.
    // - `ipiv_ptr` points to `d_ipiv`, an `n`-element i32 buffer alloc'd at
    //   line 831 — cuSOLVER writes the 1-based pivot indices (length n).
    // - `info_ptr` points to a single-element i32 alloc'd at line 832; cuSOLVER
    //   stores the status code there. We synchronize at line 869 before reading
    //   it via `read_dev_info` and branching on non-zero at line 871.
    // - All four `_sync` guards keep the `CudaSlice` borrows alive across the
    //   FFI call, ensuring no concurrent reuse of these device pointers on
    //   `stream` for the duration of the launch.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

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
            op: "gpu_lu_factor_f32: getrf failed (singular matrix)",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    // Column-major LU → row-major.
    let lu_row = crate::kernels::gpu_transpose_2d(&d_a_col, n, n, device)?;
    Ok((lu_row, d_ipiv))
}

/// GPU-resident LU factorization (f64). Mirrors [`gpu_lu_factor_f32`]. (#604)
#[cfg(feature = "cuda")]
pub fn gpu_lu_factor_f64(
    a_dev: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<i32>)> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        return Ok((
            crate::transfer::alloc_zeros_f64(0, device)?,
            alloc_zeros_i32(0, device)?,
        ));
    }
    if a_dev.len() != n * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_lu_factor_f64: input length mismatch",
            expected: vec![n * n],
            got: vec![a_dev.len()],
        });
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    let mut d_a_col = crate::kernels::gpu_transpose_2d_f64(a_dev, n, n, device)?;
    let mut d_ipiv = alloc_zeros_i32(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 908 by `cusolverDnHandle::new(stream.clone())`. The
    //   handle outlives this query because `dn` is dropped at function exit.
    // - `n` was validated at line 899-905 (early-return on length mismatch)
    //   and is non-zero (line 893 short-circuits the n==0 path).
    // - `a_ptr` points to an `n*n` f64 column-major buffer allocated at line 910
    //   by `gpu_transpose_2d_f64`; `lda = n as i32` matches the column-major
    //   leading dimension (lda >= max(1, n)) as required by the LAPACK-style ABI.
    // - `&mut lwork` is a stack-resident `i32`; cuSOLVER writes the f64
    //   workspace size in elements per the upstream `Dgetrf_bufferSize` ABI.
    // - This is a query-only call: it inspects dims, not data, so the
    //   `device_ptr_mut` borrow does not race with any other in-flight launch.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnDgetrf_bufferSize(
            dn.cu(),
            n as i32,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f64(lwork.max(1) as usize, device)?;

    // SAFETY:
    // - `dn.cu()` is the same valid handle proved above.
    // - `a_ptr` points to the `n*n` column-major f64 matrix (line 910); cuSOLVER
    //   overwrites it in place with the LU factors (L below diag, U on/above).
    // - `lda = n as i32` matches the column-major leading dimension.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f64 elements
    //   allocated at line 928 from the `Dgetrf_bufferSize` query above; the
    //   `.max(1)` guard prevents zero-sized alloc and cuSOLVER receives the
    //   queried `lwork` so the buffer is at least the size it claimed to need.
    // - `ipiv_ptr` points to `d_ipiv`, an `n`-element i32 buffer alloc'd at
    //   line 911 for the 1-based pivot indices.
    // - `info_ptr` points to a single-element i32 alloc'd at line 912; cuSOLVER
    //   writes the status code asynchronously on `stream`. We synchronize at
    //   line 949 before reading via `read_dev_info` and branching on non-zero.
    // - All four `_sync` guards keep the `CudaSlice` borrows alive for the
    //   duration of the FFI call, preventing concurrent reuse on `stream`.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDgetrf(
            dn.cu(),
            n as i32,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            work_ptr as *mut f64,
            ipiv_ptr as *mut i32,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_lu_factor_f64: getrf failed (singular matrix)",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    let lu_row = crate::kernels::gpu_transpose_2d_f64(&d_a_col, n, n, device)?;
    Ok((lu_row, d_ipiv))
}

// ---------------------------------------------------------------------------
// Device-resident Cholesky: A = L L^T  (#632)
// ---------------------------------------------------------------------------
//
// Cholesky on a symmetric positive-definite matrix only needs the lower
// (or upper) triangle. cuSOLVER `Spotrf` operates in-place; for a
// symmetric matrix the row-major view of `[i, j]` equals the column-major
// `[j, i]` so we can pass a `memcpy_dtod` clone of A directly without a
// transpose pass — the upper-triangle scratch space is overwritten by L.
//
// Then we mask the upper triangle to zero on host (small, n × n triangle
// of words) and re-upload as the row-major output L. For typical matrix
// sizes the host pass is dominated by the GPU compute time of potrf.

/// Device-resident Cholesky factorization (f32). Mirrors `gpu_cholesky_f32`
/// but accepts/returns `&CudaBuffer<f32>` — no host bounce on the
/// expensive parts. (#632)
#[cfg(feature = "cuda")]
pub fn gpu_cholesky_f32_dev(
    a_dev: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::cusolver::sys as csys;

    if a_dev.len() != n * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_cholesky_f32_dev: A length mismatch",
            expected: vec![n * n],
            got: vec![a_dev.len()],
        });
    }
    if n == 0 {
        return crate::transfer::alloc_zeros_f32(0, device);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // SPD symmetric → row-major == column-major. Just clone on device.
    let mut d_a = crate::transfer::alloc_zeros_f32(n * n, device)?;
    stream.memcpy_dtod(a_dev.inner(), d_a.inner_mut())?;

    let mut d_info = alloc_zeros_i32(1, device)?;
    let uplo = csys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;

    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 1000 by `cusolverDnHandle::new(stream.clone())`; the
    //   handle outlives this call (dn drops only at function exit).
    // - `n` was validated at line 988-993 (early-return on length mismatch)
    //   and is non-zero (line 995 short-circuits n==0).
    // - `a_ptr` points to an `n*n` f32 buffer (`d_a`) allocated at line 1003
    //   then populated via `memcpy_dtod` at line 1004 (a clone of the input).
    //   The matrix is symmetric SPD, so its row-major and column-major layouts
    //   are byte-identical (per the doc-comment at line 1002).
    // - `lda = n as i32` matches the leading dimension (lda >= max(1, n)).
    // - `uplo = CUBLAS_FILL_MODE_LOWER` (set at line 1007); cuSOLVER will
    //   write L into the lower triangle, leaving the upper triangle as
    //   workspace.
    // - `&mut lwork` is a stack-resident `i32`; cuSOLVER writes the workspace
    //   size in f32 elements per the `Spotrf_bufferSize` upstream contract.
    // - This is a query-only call: it inspects dims/uplo only, not data, so
    //   the `device_ptr_mut` borrow does not race with any other in-flight op.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnSpotrf_bufferSize(
            dn.cu(),
            uplo,
            n as i32,
            a_ptr as *mut f32,
            n as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f32(lwork.max(1) as usize, device)?;

    // SAFETY:
    // - `dn.cu()` is the valid handle proved above.
    // - `a_ptr` points to the `n*n` f32 buffer (line 1003); cuSOLVER overwrites
    //   the lower triangle with the Cholesky factor L when `uplo == LOWER`.
    // - `lda = n as i32` matches the leading dimension.
    // - `uplo = CUBLAS_FILL_MODE_LOWER` matches the value used in the
    //   buffer-size query above.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f32 elements
    //   allocated at line 1023 from the buffer-size query; cuSOLVER receives
    //   the queried `lwork`, so the buffer is at least the size it claimed.
    // - `info_ptr` points to a single-element i32 alloc'd at line 1006; the
    //   status is checked at line 1044 (`read_dev_info` after synchronize at
    //   line 1043). A non-zero info indicates the matrix is not SPD; that's
    //   converted to a `GpuError::ShapeMismatch` at line 1046.
    // - All three `_sync` guards keep the `CudaSlice` borrows alive across the
    //   FFI call so cudarc doesn't reuse these device pointers on `stream`.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnSpotrf(
            dn.cu(),
            uplo,
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
            op: "gpu_cholesky_f32_dev: potrf failed (matrix not positive definite)",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    // potrf with UPLO_LOWER stores L in the lower triangle of A
    // (column-major). For symmetric input, the row-major lower triangle
    // is the column-major lower triangle's element-wise mirror — we
    // need to zero the upper triangle in row-major view. Mask on host
    // (O(n^2) small).
    let host = crate::transfer::gpu_to_cpu(&d_a, device)?;
    let mut row_major = vec![0.0_f32; n * n];
    // The data on device is column-major n×n with L in lower triangle.
    // Convert column-major (j, i) → row-major (i, j) and zero the upper
    // triangle in row-major (i.e. j > i).
    for j in 0..n {
        for i in 0..n {
            let cm = j * n + i; // column-major (i, j)
            if i >= j {
                // Lower-or-diag in column-major = lower-or-diag in row-major
                // when transposed: we want row-major (i, j) for i >= j.
                row_major[i * n + j] = host[cm];
            }
        }
    }
    crate::transfer::cpu_to_gpu(&row_major, device)
}

/// f64 device-resident Cholesky. (#632)
#[cfg(feature = "cuda")]
pub fn gpu_cholesky_f64_dev(
    a_dev: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::cusolver::sys as csys;

    if a_dev.len() != n * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_cholesky_f64_dev: A length mismatch",
            expected: vec![n * n],
            got: vec![a_dev.len()],
        });
    }
    if n == 0 {
        return crate::transfer::alloc_zeros_f64(0, device);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    let mut d_a = crate::transfer::alloc_zeros_f64(n * n, device)?;
    stream.memcpy_dtod(a_dev.inner(), d_a.inner_mut())?;

    let mut d_info = alloc_zeros_i32(1, device)?;
    let uplo = csys::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;

    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 1097 by `cusolverDnHandle::new(stream.clone())`; the
    //   handle outlives this call (dn drops only at function exit).
    // - `n` was validated at line 1085-1090 (early-return on length mismatch)
    //   and is non-zero (line 1092 short-circuits the n==0 path).
    // - `a_ptr` points to an `n*n` f64 buffer (`d_a`) allocated at line 1099
    //   and populated via `memcpy_dtod` at line 1100. Per the symmetric-SPD
    //   invariant (doc-comment at line 1002 of the f32 sibling), row-major
    //   and column-major layouts are byte-identical for the input.
    // - `lda = n as i32` matches the leading dimension (lda >= max(1, n)).
    // - `uplo = CUBLAS_FILL_MODE_LOWER` (set at line 1103); cuSOLVER will
    //   write L into the lower triangle.
    // - `&mut lwork` is stack-resident `i32`; cuSOLVER writes the f64
    //   workspace size in elements per the `Dpotrf_bufferSize` ABI.
    // - This is a query-only call: it inspects dims/uplo only, not the data.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnDpotrf_bufferSize(
            dn.cu(),
            uplo,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f64(lwork.max(1) as usize, device)?;

    // SAFETY:
    // - `dn.cu()` is the valid handle proved above.
    // - `a_ptr` points to the `n*n` f64 buffer (line 1099); cuSOLVER overwrites
    //   the lower triangle with the Cholesky factor L when uplo == LOWER.
    // - `lda = n as i32` matches the leading dimension.
    // - `uplo = CUBLAS_FILL_MODE_LOWER` matches the buffer-size query.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f64 elements
    //   allocated at line 1119 from the `Dpotrf_bufferSize` query above; the
    //   `.max(1)` guard prevents zero-sized alloc and cuSOLVER receives the
    //   queried `lwork` so the buffer is at least the size it claimed to need.
    // - `info_ptr` points to a single-element i32 alloc'd at line 1102; the
    //   status is checked at line 1140 (`read_dev_info` after synchronize at
    //   line 1139) and converted to `GpuError::ShapeMismatch` on non-zero
    //   (positive info_val ⇒ leading minor not positive definite).
    // - All three `_sync` guards keep the `CudaSlice` borrows alive across the
    //   FFI call so cudarc cannot reuse these device pointers on `stream`.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDpotrf(
            dn.cu(),
            uplo,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            work_ptr as *mut f64,
            lwork,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_cholesky_f64_dev: potrf failed (matrix not positive definite)",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    let host = crate::transfer::gpu_to_cpu(&d_a, device)?;
    let mut row_major = vec![0.0_f64; n * n];
    for j in 0..n {
        for i in 0..n {
            let cm = j * n + i;
            if i >= j {
                row_major[i * n + j] = host[cm];
            }
        }
    }
    crate::transfer::cpu_to_gpu(&row_major, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_cholesky_f32_dev(
    _a: &CudaBuffer<f32>,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_cholesky_f64_dev(
    _a: &CudaBuffer<f64>,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Device-resident solve: A * X = B   (#632)
// ---------------------------------------------------------------------------
//
// `gpu_solve_f32_dev` / `gpu_solve_f64_dev` mirror `gpu_solve_f32` /
// `gpu_solve_f64` but accept `&CudaBuffer<T>` instead of `&[T]` and return
// `CudaBuffer<T>` instead of `Vec<T>`. The row→column-major transpose is
// done on device via `gpu_transpose_2d`, eliminating the host bounce that
// the API-boundary ops needed.

/// Device-resident solve `A * X = B` (f32). A is row-major n×n; B is
/// row-major n×nrhs. Returns X as a row-major device buffer of length
/// `n * nrhs`. (#632)
#[cfg(feature = "cuda")]
pub fn gpu_solve_f32_dev(
    a_dev: &CudaBuffer<f32>,
    b_dev: &CudaBuffer<f32>,
    n: usize,
    nrhs: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::cusolver::sys as csys;

    if a_dev.len() != n * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f32_dev: A length mismatch",
            expected: vec![n * n],
            got: vec![a_dev.len()],
        });
    }
    if b_dev.len() != n * nrhs {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f32_dev: B length mismatch",
            expected: vec![n * nrhs],
            got: vec![b_dev.len()],
        });
    }
    if n == 0 {
        return crate::transfer::alloc_zeros_f32(0, device);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // On-device transposes: row → column-major.
    let mut d_a = crate::kernels::gpu_transpose_2d(a_dev, n, n, device)?;
    let mut d_b = crate::kernels::gpu_transpose_2d(b_dev, n, nrhs, device)?;

    let mut d_ipiv = alloc_zeros_i32(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 1221 by `cusolverDnHandle::new(stream.clone())`; the
    //   handle outlives this call (dn drops at function exit).
    // - `n` was validated at line 1202-1207 (early-return on `a_dev.len() != n*n`)
    //   and is non-zero (line 1216 short-circuits the n==0 path).
    // - `a_ptr` points to an `n*n` f32 column-major buffer (`d_a`) produced by
    //   `gpu_transpose_2d` at line 1224 (row→col on device).
    // - `lda = n as i32` matches the column-major leading dimension; cuSOLVER's
    //   LAPACK-style ABI requires `lda >= max(1, n)`.
    // - `&mut lwork` is stack-resident `i32`; cuSOLVER writes the workspace
    //   size in f32 elements per the `Sgetrf_bufferSize` upstream contract.
    // - This is a query-only call: it inspects dims only, not data.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
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

    // SAFETY:
    // - `dn.cu()` is the valid handle proved above.
    // - `a_ptr` points to the column-major `n*n` f32 matrix (line 1224); cuSOLVER
    //   overwrites it in place with the LU factors.
    // - `lda = n as i32` matches the column-major leading dimension.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f32 elements
    //   allocated at line 1244 from the `Sgetrf_bufferSize` query above; the
    //   `.max(1)` prevents a zero-sized alloc and cuSOLVER receives the
    //   queried `lwork` so the buffer is at least the size it claimed.
    // - `ipiv_ptr` points to `d_ipiv`, an `n`-element i32 buffer alloc'd at
    //   line 1227 for the 1-based pivot indices (length n).
    // - `info_ptr` points to a single-element i32 alloc'd at line 1228; the
    //   status is checked at line 1266 (`read_dev_info` after synchronize at
    //   line 1265). A non-zero info_val here indicates a singular matrix and
    //   is converted to `GpuError::ShapeMismatch` at line 1268.
    // - All four `_sync` guards keep the `CudaSlice` borrows alive across the
    //   FFI call.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

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
            op: "gpu_solve_f32_dev: LU factorization failed (singular matrix)",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    let mut d_info2 = alloc_zeros_i32(1, device)?;

    // SAFETY:
    // - `dn.cu()` is the same valid handle (still alive).
    // - The Sgetrf call above succeeded (we checked devInfo at line 1266 and
    //   returned early on non-zero), so `d_a` now contains valid LU factors
    //   and `d_ipiv` valid pivot indices — both are read-only here, so we
    //   take immutable `device_ptr` borrows (cast to `*const f32` / `*const i32`).
    // - `b_ptr` points to `d_b`, an `n*nrhs` column-major f32 buffer produced
    //   by `gpu_transpose_2d` at line 1225; cuSOLVER overwrites it in place
    //   with the solution X (still column-major).
    // - `nrhs` was validated at line 1209-1214 (B length == n*nrhs).
    // - `lda = ldb = n as i32` for both the LU matrix and the RHS — both are
    //   leading dimensions of square / n-row column-major matrices.
    // - `op = CUBLAS_OP_N` (no transpose); we factored A directly, so we
    //   solve A * X = B not A^T * X = B.
    // - `info_ptr` points to a fresh i32 alloc (`d_info2` at line 1275),
    //   distinct from the getrf info to make failure mode unambiguous.
    //   Status is checked at line 1299 after synchronize.
    // - All four `_sync` guards keep the `CudaSlice` borrows alive across
    //   the FFI call.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner().device_ptr(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner().device_ptr(&stream);
        let (b_ptr, _b_sync) = d_b.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info2.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnSgetrs(
            dn.cu(),
            csys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            nrhs as i32,
            a_ptr as *const f32,
            n as i32,
            ipiv_ptr as *const i32,
            b_ptr as *mut f32,
            n as i32,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info2 = read_dev_info(&d_info2, device)?;
    if info2 != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f32_dev: triangular solve failed",
            expected: vec![0],
            got: vec![info2 as usize],
        });
    }

    // Solution X is column-major n×nrhs in d_b; transpose to row-major.
    crate::kernels::gpu_transpose_2d(&d_b, nrhs, n, device)
}

/// f64 device-resident solve. Mirrors [`gpu_solve_f32_dev`]. (#632)
#[cfg(feature = "cuda")]
pub fn gpu_solve_f64_dev(
    a_dev: &CudaBuffer<f64>,
    b_dev: &CudaBuffer<f64>,
    n: usize,
    nrhs: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::cusolver::sys as csys;

    if a_dev.len() != n * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f64_dev: A length mismatch",
            expected: vec![n * n],
            got: vec![a_dev.len()],
        });
    }
    if b_dev.len() != n * nrhs {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f64_dev: B length mismatch",
            expected: vec![n * nrhs],
            got: vec![b_dev.len()],
        });
    }
    if n == 0 {
        return crate::transfer::alloc_zeros_f64(0, device);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    let mut d_a = crate::kernels::gpu_transpose_2d_f64(a_dev, n, n, device)?;
    let mut d_b = crate::kernels::gpu_transpose_2d_f64(b_dev, n, nrhs, device)?;
    let mut d_ipiv = alloc_zeros_i32(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 1342 by `cusolverDnHandle::new(stream.clone())`; the
    //   handle outlives this call (dn drops at function exit).
    // - `n` was validated at line 1323-1328 (early-return on length mismatch)
    //   and is non-zero (line 1337 short-circuits the n==0 path).
    // - `a_ptr` points to an `n*n` f64 column-major buffer (`d_a`) produced by
    //   `gpu_transpose_2d_f64` at line 1344 (row→col on device).
    // - `lda = n as i32` matches the column-major leading dimension; cuSOLVER's
    //   LAPACK-style ABI requires `lda >= max(1, n)`.
    // - `&mut lwork` is stack-resident `i32`; cuSOLVER writes the f64
    //   workspace size in elements per the `Dgetrf_bufferSize` ABI.
    // - This is a query-only call: it inspects dims only, not data.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnDgetrf_bufferSize(
            dn.cu(),
            n as i32,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f64(lwork.max(1) as usize, device)?;

    // SAFETY:
    // - `dn.cu()` is the valid handle proved above.
    // - `a_ptr` points to the column-major `n*n` f64 matrix (line 1344);
    //   cuSOLVER overwrites it in place with the LU factors.
    // - `lda = n as i32` matches the column-major leading dimension.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f64 elements
    //   allocated at line 1363 from the buffer-size query above; the `.max(1)`
    //   prevents zero-sized alloc and cuSOLVER receives the queried `lwork`.
    // - `ipiv_ptr` points to `d_ipiv`, an `n`-element i32 buffer alloc'd at
    //   line 1346 for the 1-based pivot indices (length n).
    // - `info_ptr` points to a single-element i32 alloc'd at line 1347;
    //   status is checked at line 1385 (`read_dev_info` after synchronize at
    //   line 1384), and non-zero info is converted to `GpuError::ShapeMismatch`
    //   (positive ⇒ singular U_kk == 0 at column k).
    // - All four `_sync` guards keep the `CudaSlice` borrows alive across the
    //   FFI call.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDgetrf(
            dn.cu(),
            n as i32,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            work_ptr as *mut f64,
            ipiv_ptr as *mut i32,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f64_dev: LU factorization failed",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    let mut d_info2 = alloc_zeros_i32(1, device)?;

    // SAFETY:
    // - `dn.cu()` is the same valid handle (still alive).
    // - The Dgetrf call above succeeded (we checked devInfo at line 1385 and
    //   returned early on non-zero), so `d_a` contains valid LU factors and
    //   `d_ipiv` valid pivot indices — both are read-only here, so we take
    //   immutable `device_ptr` borrows (cast to `*const f64` / `*const i32`).
    // - `b_ptr` points to `d_b`, an `n*nrhs` column-major f64 buffer produced
    //   by `gpu_transpose_2d_f64` at line 1345; cuSOLVER overwrites it in
    //   place with the solution X (still column-major).
    // - `nrhs` was validated at line 1330-1335 (B length == n*nrhs).
    // - `lda = ldb = n as i32` for both the LU matrix and the RHS — both are
    //   leading dimensions of square/n-row column-major matrices.
    // - `op = CUBLAS_OP_N` (no transpose); we factored A directly, solving
    //   A * X = B.
    // - `info_ptr` points to a fresh i32 alloc (`d_info2` at line 1394),
    //   distinct from the getrf info; status is checked at line 1418 after
    //   synchronize.
    // - All four `_sync` guards keep the `CudaSlice` borrows alive across
    //   the FFI call.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner().device_ptr(&stream);
        let (ipiv_ptr, _ipiv_sync) = d_ipiv.inner().device_ptr(&stream);
        let (b_ptr, _b_sync) = d_b.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info2.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDgetrs(
            dn.cu(),
            csys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            nrhs as i32,
            a_ptr as *const f64,
            n as i32,
            ipiv_ptr as *const i32,
            b_ptr as *mut f64,
            n as i32,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info2 = read_dev_info(&d_info2, device)?;
    if info2 != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_solve_f64_dev: triangular solve failed",
            expected: vec![0],
            got: vec![info2 as usize],
        });
    }

    crate::kernels::gpu_transpose_2d_f64(&d_b, nrhs, n, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_solve_f32_dev(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _n: usize,
    _nrhs: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}
#[cfg(not(feature = "cuda"))]
pub fn gpu_solve_f64_dev(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _n: usize,
    _nrhs: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Least-squares: solve min ||A x - b||  via cusolverDn{SS,DD}gels (#630)
// ---------------------------------------------------------------------------
//
// Cudarc only exposes the iterative-refinement variants (SSgels, DDgels);
// these use the same precision as the input dtype throughout. Both expect
// column-major operands. We use the existing gpu_transpose_2d /
// gpu_transpose_2d_f64 to flip row-major → column-major on device.
//
// API: takes A (m×n) and B (m×nrhs) as device buffers, returns X (n×nrhs)
// as a device buffer — fully on-device, no host bounce.

/// GPU-resident least-squares solver (f32). Mirrors `torch.linalg.lstsq`'s
/// solution output. Returns X minimizing `||A X - B||_F`. (#630)
#[cfg(feature = "cuda")]
pub fn gpu_lstsq_f32(
    a_dev: &CudaBuffer<f32>,
    b_dev: &CudaBuffer<f32>,
    m: usize,
    n: usize,
    nrhs: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::cusolver::sys as csys;

    if a_dev.len() != m * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_lstsq_f32: A length mismatch",
            expected: vec![m * n],
            got: vec![a_dev.len()],
        });
    }
    if b_dev.len() != m * nrhs {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_lstsq_f32: B length mismatch",
            expected: vec![m * nrhs],
            got: vec![b_dev.len()],
        });
    }
    if m == 0 || n == 0 || nrhs == 0 {
        return crate::transfer::alloc_zeros_f32(n * nrhs, device);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Row-major → column-major on device.
    let mut d_a_col = crate::kernels::gpu_transpose_2d(a_dev, m, n, device)?;
    let mut d_b_col = crate::kernels::gpu_transpose_2d(b_dev, m, nrhs, device)?;

    let mut d_x = crate::transfer::alloc_zeros_f32(n * nrhs, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;
    let mut iter: i32 = 0;

    // Query workspace.
    let mut lwork_bytes: usize = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 1495 by `cusolverDnHandle::new(stream.clone())`.
    // - `m`, `n`, `nrhs` were validated at lines 1476-1492 (A length == m*n,
    //   B length == m*nrhs, all dims non-zero).
    // - `a_ptr` points to an `m*n` f32 column-major buffer alloc'd at line 1498
    //   by `gpu_transpose_2d`; `lda = m as i32` matches the column-major
    //   leading dimension (cuSOLVER requires `lda >= max(1, m)`).
    // - `b_ptr` points to an `m*nrhs` f32 column-major buffer alloc'd at line
    //   1499 by `gpu_transpose_2d`; `ldb = m as i32` matches `max(1, m)`.
    // - `x_ptr` points to an `n*nrhs` f32 buffer alloc'd zero-init at line
    //   1501; `ldx = n as i32` matches the column-major leading dim
    //   (`ldx >= max(1, n)`); cuSOLVER will write the solution X here.
    // - The 11th argument is `std::ptr::null_mut()`: per cudarc/cuSOLVER docs,
    //   `niters_ptr` is unused for the buffer-size query and accepts NULL.
    // - `&mut lwork_bytes` is stack-resident `usize`; cuSOLVER writes the
    //   workspace size **in bytes** (not elements) per the SSgels ABI,
    //   distinguishing it from the `Sgetrf_bufferSize` lwork-in-elements
    //   convention used elsewhere in this file.
    // - This is a query-only call: it inspects dims only, not data, so the
    //   `device_ptr_mut` borrows do not race with any concurrent launch.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (b_ptr, _b_sync) = d_b_col.inner_mut().device_ptr_mut(&stream);
        let (x_ptr, _x_sync) = d_x.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnSSgels_bufferSize(
            dn.cu(),
            m as i32,
            n as i32,
            nrhs as i32,
            a_ptr as *mut f32,
            m as i32,
            b_ptr as *mut f32,
            m as i32,
            x_ptr as *mut f32,
            n as i32,
            std::ptr::null_mut(),
            &mut lwork_bytes,
        )
        .result()?;
    }

    let work_elems = lwork_bytes.div_ceil(4).max(1);
    let mut d_work = crate::transfer::alloc_zeros_f32(work_elems, device)?;

    // SAFETY:
    // - `dn.cu()` is the valid handle proved above.
    // - `a_ptr` points to the `m*n` f32 column-major matrix (line 1498); SSgels
    //   may overwrite it in place during iterative refinement.
    // - `b_ptr` points to the `m*nrhs` f32 column-major RHS (line 1499); SSgels
    //   may overwrite it.
    // - `x_ptr` points to the `n*nrhs` f32 output buffer (line 1501); cuSOLVER
    //   writes the solution X here in column-major layout.
    // - `lda = ldb = m as i32` and `ldx = n as i32` — matches column-major
    //   leading dimensions for matrices with `m` or `n` rows respectively.
    // - `work_ptr` points to a workspace of `work_elems` f32 elements
    //   allocated at line 1529; `work_elems = lwork_bytes.div_ceil(4).max(1)`
    //   computes the number of f32 (4-byte) elements needed to cover the
    //   queried `lwork_bytes`, with `.max(1)` guarding against zero-sized
    //   alloc. cuSOLVER receives the original `lwork_bytes` count, so it sees
    //   exactly the byte budget it asked for.
    // - The work pointer is cast to `*mut std::ffi::c_void` per the SSgels ABI
    //   (which uses an opaque byte-buffer signature, not f32-typed).
    // - `&mut iter` is a stack-resident `i32` (at line 1503); cuSOLVER writes
    //   the iterative-refinement step count here. It's reported via the
    //   `info_ptr` channel for failure modes; we don't currently inspect iter.
    // - `info_ptr` points to a single-element i32 alloc'd at line 1502; the
    //   status is checked at line 1558 (`read_dev_info` after synchronize at
    //   line 1557). Negative info_val ⇒ invalid argument; positive info_val
    //   in this routine indicates iter-refinement non-convergence which
    //   cudarc encodes via the negative-only branch at line 1559.
    // - All five `_sync` guards keep the `CudaSlice` borrows alive across
    //   the FFI call.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (b_ptr, _b_sync) = d_b_col.inner_mut().device_ptr_mut(&stream);
        let (x_ptr, _x_sync) = d_x.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnSSgels(
            dn.cu(),
            m as i32,
            n as i32,
            nrhs as i32,
            a_ptr as *mut f32,
            m as i32,
            b_ptr as *mut f32,
            m as i32,
            x_ptr as *mut f32,
            n as i32,
            work_ptr as *mut std::ffi::c_void,
            lwork_bytes,
            &mut iter,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info_val = read_dev_info(&d_info, device)?;
    if info_val < 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_lstsq_f32: SSgels reported an invalid argument",
            expected: vec![0],
            got: vec![info_val.unsigned_abs() as usize],
        });
    }

    // Solution X is column-major n×nrhs; convert back to row-major.
    crate::kernels::gpu_transpose_2d(&d_x, n, nrhs, device)
}

/// GPU-resident least-squares solver (f64). Mirrors [`gpu_lstsq_f32`]. (#630)
#[cfg(feature = "cuda")]
pub fn gpu_lstsq_f64(
    a_dev: &CudaBuffer<f64>,
    b_dev: &CudaBuffer<f64>,
    m: usize,
    n: usize,
    nrhs: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::cusolver::sys as csys;

    if a_dev.len() != m * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_lstsq_f64: A length mismatch",
            expected: vec![m * n],
            got: vec![a_dev.len()],
        });
    }
    if b_dev.len() != m * nrhs {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_lstsq_f64: B length mismatch",
            expected: vec![m * nrhs],
            got: vec![b_dev.len()],
        });
    }
    if m == 0 || n == 0 || nrhs == 0 {
        return crate::transfer::alloc_zeros_f64(n * nrhs, device);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    let mut d_a_col = crate::kernels::gpu_transpose_2d_f64(a_dev, m, n, device)?;
    let mut d_b_col = crate::kernels::gpu_transpose_2d_f64(b_dev, m, nrhs, device)?;
    let mut d_x = crate::transfer::alloc_zeros_f64(n * nrhs, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;
    let mut iter: i32 = 0;

    let mut lwork_bytes: usize = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 1602 by `cusolverDnHandle::new(stream.clone())`.
    // - `m`, `n`, `nrhs` were validated at lines 1583-1599 (A length == m*n,
    //   B length == m*nrhs, all dims non-zero).
    // - `a_ptr` points to an `m*n` f64 column-major buffer alloc'd at line
    //   1604 by `gpu_transpose_2d_f64`; `lda = m as i32` matches column-major
    //   leading dim (`lda >= max(1, m)`).
    // - `b_ptr` points to an `m*nrhs` f64 column-major RHS alloc'd at line
    //   1605; `ldb = m as i32`.
    // - `x_ptr` points to an `n*nrhs` f64 output buffer alloc'd zero-init at
    //   line 1606; `ldx = n as i32` (`ldx >= max(1, n)`).
    // - The 11th argument is `std::ptr::null_mut()`: `niters_ptr` is unused
    //   for the buffer-size query and accepts NULL per cudarc/cuSOLVER docs.
    // - `&mut lwork_bytes` is stack-resident `usize`; cuSOLVER writes the
    //   workspace size **in bytes** per the DDgels ABI (distinct from
    //   `Dgetrf_bufferSize`'s elements convention).
    // - This is a query-only call: it inspects dims only, not data.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (b_ptr, _b_sync) = d_b_col.inner_mut().device_ptr_mut(&stream);
        let (x_ptr, _x_sync) = d_x.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnDDgels_bufferSize(
            dn.cu(),
            m as i32,
            n as i32,
            nrhs as i32,
            a_ptr as *mut f64,
            m as i32,
            b_ptr as *mut f64,
            m as i32,
            x_ptr as *mut f64,
            n as i32,
            std::ptr::null_mut(),
            &mut lwork_bytes,
        )
        .result()?;
    }

    let work_elems = lwork_bytes.div_ceil(8).max(1);
    let mut d_work = crate::transfer::alloc_zeros_f64(work_elems, device)?;

    // SAFETY:
    // - `dn.cu()` is the valid handle proved above.
    // - `a_ptr` (line 1604), `b_ptr` (line 1605), `x_ptr` (line 1606) all
    //   point to column-major buffers whose sizes were validated above.
    //   DDgels may overwrite A and B in place during iterative refinement;
    //   X receives the solution.
    // - `lda = ldb = m as i32` and `ldx = n as i32` — column-major leading
    //   dimensions matching their respective row counts.
    // - `work_ptr` points to a workspace of `work_elems` f64 elements
    //   allocated at line 1633; `work_elems = lwork_bytes.div_ceil(8).max(1)`
    //   computes the count of f64 (8-byte) elements covering the queried
    //   `lwork_bytes` (the `.div_ceil(8)` matches the f64 size; `.max(1)`
    //   prevents zero-sized alloc). cuSOLVER receives the original byte count
    //   `lwork_bytes`, so the buffer is at least that many bytes.
    // - The work pointer is cast to `*mut std::ffi::c_void` per the DDgels
    //   ABI's opaque-byte-buffer signature.
    // - `&mut iter` is a stack-resident `i32` (line 1608); cuSOLVER writes
    //   the iterative-refinement step count there.
    // - `info_ptr` points to a single-element i32 alloc'd at line 1607; status
    //   is checked at line 1662 (`read_dev_info` after synchronize at line
    //   1661). Negative info_val ⇒ invalid argument.
    // - All five `_sync` guards keep the `CudaSlice` borrows alive across
    //   the FFI call.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (b_ptr, _b_sync) = d_b_col.inner_mut().device_ptr_mut(&stream);
        let (x_ptr, _x_sync) = d_x.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDDgels(
            dn.cu(),
            m as i32,
            n as i32,
            nrhs as i32,
            a_ptr as *mut f64,
            m as i32,
            b_ptr as *mut f64,
            m as i32,
            x_ptr as *mut f64,
            n as i32,
            work_ptr as *mut std::ffi::c_void,
            lwork_bytes,
            &mut iter,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info_val = read_dev_info(&d_info, device)?;
    if info_val < 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_lstsq_f64: DDgels reported an invalid argument",
            expected: vec![0],
            got: vec![info_val.unsigned_abs() as usize],
        });
    }

    crate::kernels::gpu_transpose_2d_f64(&d_x, n, nrhs, device)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_lstsq_f32(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _m: usize,
    _n: usize,
    _nrhs: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_lstsq_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _m: usize,
    _n: usize,
    _nrhs: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Non-symmetric eigendecomposition: A = V Λ V^{-1}  via cusolverDnXgeev (#631)
// ---------------------------------------------------------------------------
//
// Wraps the cusolver "X" (extended) geev API. For real input A of size n×n:
//   - W:  complex eigenvalues, length n, stored as `[n, 2]` interleaved
//         re/im pairs (matches the FFT convention).
//   - VR: complex right eigenvectors, n×n, stored as `[n, n, 2]` interleaved.
// We compute right eigenvectors only (jobvl = NOVECTOR, jobvr = VECTOR) since
// torch.linalg.eig also returns just (eigenvalues, eigenvectors).

#[cfg(feature = "cuda")]
struct DnParamsHandle {
    inner: cudarc::cusolver::sys::cusolverDnParams_t,
}

#[cfg(feature = "cuda")]
impl DnParamsHandle {
    fn new() -> GpuResult<Self> {
        use cudarc::cusolver::sys as csys;
        let mut p: csys::cusolverDnParams_t = std::ptr::null_mut();
        // SAFETY:
        // - `cusolverDnCreateParams` takes a `*mut cusolverDnParams_t` and
        //   writes a freshly-allocated handle into the slot pointed at.
        // - `&mut p` is a valid pointer to a stack-local `cusolverDnParams_t`
        //   (declared on the previous line as a typed null pointer); the
        //   pointee lifetime covers this FFI call (alloc happens before the
        //   `Ok(Self { inner: p })` move on the next line).
        // - On error, cuSOLVER may leave `p` as the original null value;
        //   `.result()?` propagates the error before we wrap `p` in `Self`,
        //   so we never construct a `DnParamsHandle` containing a sentinel
        //   that could later be passed to a routine call.
        // - On success, `p` becomes a valid handle that will be destroyed
        //   exactly once via the `Drop` impl below — matching the cuSOLVER
        //   create/destroy contract.
        // - This call has no device-side side effects: it only allocates a
        //   small CPU-side opaque struct and is safe to make even before the
        //   stream is bound.
        unsafe {
            csys::cusolverDnCreateParams(&mut p).result()?;
        }
        Ok(Self { inner: p })
    }
    fn raw(&self) -> cudarc::cusolver::sys::cusolverDnParams_t {
        self.inner
    }
}

#[cfg(feature = "cuda")]
impl Drop for DnParamsHandle {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            // SAFETY:
            // - `self.inner` is non-null (just checked at line 1733) and was
            //   produced by `cusolverDnCreateParams` in `Self::new` (line
            //   1727); per the cuSOLVER contract, every successful Create
            //   must be paired with exactly one Destroy.
            // - `Drop::drop` runs at most once per `DnParamsHandle` value
            //   (the Rust drop guarantee), so `Destroy` is never invoked
            //   twice on the same handle — no double-free.
            // - We discard the returned status (`let _ = ...`) because Drop
            //   cannot return errors and we cannot panic-propagate them
            //   safely from a destructor; the params object is a small CPU
            //   struct with no externally-observable cleanup beyond memory
            //   release, so a leaked failure would manifest only as a tiny
            //   leak, not a correctness bug.
            // - No `cusolverDnHandle_t` or `stream` is needed here:
            //   `cusolverDnDestroyParams` operates only on the params object.
            unsafe {
                let _ = cudarc::cusolver::sys::cusolverDnDestroyParams(self.inner);
            }
        }
    }
}

/// GPU-resident non-symmetric eigendecomposition (f32 → complex output).
/// Mirrors `torch.linalg.eig` for real f32 inputs. Returns
/// `(W: CudaBuffer<f32>, VR: CudaBuffer<f32>)` where:
///   - W has length `2n` interleaved re/im (logical shape `[n, 2]`)
///   - VR has length `2 * n * n` row-major (logical shape `[n, n, 2]`)
///
/// (#631)
#[cfg(feature = "cuda")]
pub fn gpu_eig_f32(
    a_dev: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    use cudarc::cusolver::sys as csys;

    if a_dev.len() != n * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_eig_f32: A length mismatch",
            expected: vec![n * n],
            got: vec![a_dev.len()],
        });
    }
    if n == 0 {
        return Ok((
            crate::transfer::alloc_zeros_f32(0, device)?,
            crate::transfer::alloc_zeros_f32(0, device)?,
        ));
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;
    let params = DnParamsHandle::new()?;

    // Row-major → column-major on device (cuSOLVER expects column-major).
    let mut d_a_col = crate::kernels::gpu_transpose_2d(a_dev, n, n, device)?;
    // W: complex eigenvalues, 2n f32 (re/im interleaved).
    let mut d_w = crate::transfer::alloc_zeros_f32(2 * n, device)?;
    // VR: complex right eigenvectors, 2 * n * n f32 (re/im interleaved per element).
    let mut d_vr = crate::transfer::alloc_zeros_f32(2 * n * n, device)?;
    // VL: dummy 1-element placeholder for the buffer-size argument-validation
    // path. Even with jobvl = NOVECTOR, cuSOLVER's X-API rejects null VL.
    let mut d_vl_dummy = crate::transfer::alloc_zeros_f32(2, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    let novec = csys::cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR;
    let vec_mode = csys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
    let dt_a = csys::cudaDataType::CUDA_R_32F;
    let dt_w = csys::cudaDataType::CUDA_C_32F;
    let dt_v = csys::cudaDataType::CUDA_C_32F;
    let compute_type = csys::cudaDataType::CUDA_R_32F;

    // Buffer-size query.
    let mut wks_dev_bytes: usize = 0;
    let mut wks_host_bytes: usize = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 1770 by `cusolverDnHandle::new(stream.clone())`.
    // - `params.raw()` returns the `cusolverDnParams_t` created at line 1771;
    //   the params handle is owned by the local `params` binding which lives
    //   for the entire function (Drop runs at function exit, after this call).
    // - `n` was validated at line 1755-1760 (A length == n*n) and is non-zero
    //   (line 1762 short-circuits the n==0 path).
    // - `a_ptr` points to an `n*n` f32 column-major buffer (`d_a_col`)
    //   produced by `gpu_transpose_2d` at line 1774; `lda = n as i64` matches
    //   the column-major leading dimension. Cast to `*const c_void` per the
    //   Xgeev type-erased ABI; `dt_a = CUDA_R_32F` (line 1786) tells cuSOLVER
    //   to interpret these bytes as f32.
    // - `w_ptr` points to a `2n` f32 buffer alloc'd at line 1776 holding
    //   complex eigenvalues as interleaved re/im pairs; `dt_w = CUDA_C_32F`
    //   (line 1787) matches the interleaved layout: each complex element
    //   spans 2 consecutive f32s.
    // - `vl_ptr` points to a 2-element placeholder buffer alloc'd at line
    //   1781. cuSOLVER's X-API rejects null VL even with `jobvl = NOVECTOR`,
    //   per the comment at lines 1779-1781; `ldvl = n as i64` is the leading
    //   dim. The buffer is never written to (NOVECTOR mode).
    // - `vr_ptr` points to a `2*n*n` f32 buffer alloc'd at line 1778 (complex
    //   eigenvectors interleaved); `ldvr = n as i64`; `dt_v = CUDA_C_32F`.
    // - `compute_type = CUDA_R_32F` (line 1789) — cuSOLVER's mixed-precision
    //   API requires a separate compute precision; we keep it real f32.
    // - `&mut wks_dev_bytes` and `&mut wks_host_bytes` are stack-resident
    //   `usize`; cuSOLVER writes the device and host workspace sizes
    //   **in bytes** per the Xgeev ABI's two-buffer scheme.
    // - This is a query-only call: the buffers are not written. All
    //   `device_ptr_mut` borrows are taken to satisfy cudarc's ownership
    //   bookkeeping but are not actually dereferenced for write.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (w_ptr, _w_sync) = d_w.inner_mut().device_ptr_mut(&stream);
        let (vr_ptr, _vr_sync) = d_vr.inner_mut().device_ptr_mut(&stream);
        let (vl_ptr, _vl_sync) = d_vl_dummy.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnXgeev_bufferSize(
            dn.cu(),
            params.raw(),
            novec,
            vec_mode,
            n as i64,
            dt_a,
            a_ptr as *const std::ffi::c_void,
            n as i64,
            dt_w,
            w_ptr as *const std::ffi::c_void,
            dt_v,
            vl_ptr as *const std::ffi::c_void,
            n as i64,
            dt_v,
            vr_ptr as *const std::ffi::c_void,
            n as i64,
            compute_type,
            &mut wks_dev_bytes,
            &mut wks_host_bytes,
        )
        .result()?;
    }

    let dev_work_elems = wks_dev_bytes.div_ceil(4).max(1);
    let mut d_work = crate::transfer::alloc_zeros_f32(dev_work_elems, device)?;
    let mut host_work: Vec<u8> = vec![0u8; wks_host_bytes];

    // SAFETY:
    // - `dn.cu()`, `params.raw()` are the same valid handles from above.
    // - `a_ptr` points to the `n*n` f32 column-major matrix (line 1774);
    //   Xgeev may overwrite it during reduction to Hessenberg form.
    // - `w_ptr` points to the `2n` f32 eigenvalue buffer (line 1776);
    //   cuSOLVER writes complex eigenvalues here as re/im pairs.
    // - `vl_ptr` points to the dummy 2-element VL buffer (line 1781) — not
    //   written to in NOVECTOR mode but required to be non-null by the X-API.
    // - `vr_ptr` points to the `2*n*n` f32 eigenvector buffer (line 1778);
    //   cuSOLVER writes the right eigenvectors here in column-major complex
    //   layout (each element occupies 2 consecutive f32s).
    // - `lda = ldvl = ldvr = n as i64` — column-major leading dims.
    // - `work_ptr` points to a device workspace of `dev_work_elems` f32
    //   elements alloc'd at line 1824; `dev_work_elems =
    //   wks_dev_bytes.div_ceil(4).max(1)` covers the queried byte budget
    //   (4-byte f32 elements, with `.max(1)` guarding zero-size). cuSOLVER
    //   receives `wks_dev_bytes` directly so it sees the original byte count.
    // - `host_work.as_mut_ptr()` points to a `wks_host_bytes` `u8` buffer
    //   alloc'd at line 1825 (`Vec<u8>`); cuSOLVER writes host-side scratch
    //   here. The vector outlives the FFI call (drops at function exit).
    // - `info_ptr` points to a single-element i32 alloc'd at line 1782; the
    //   status is checked at line 1863 (`read_dev_info` after synchronize at
    //   line 1862). Non-zero info means non-converged eigenvalues; converted
    //   to `GpuError::ShapeMismatch` at line 1865.
    // - All six `_sync` guards keep the `CudaSlice` borrows alive across the
    //   FFI call, preventing concurrent reuse on `stream`.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (w_ptr, _w_sync) = d_w.inner_mut().device_ptr_mut(&stream);
        let (vr_ptr, _vr_sync) = d_vr.inner_mut().device_ptr_mut(&stream);
        let (vl_ptr, _vl_sync) = d_vl_dummy.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnXgeev(
            dn.cu(),
            params.raw(),
            novec,
            vec_mode,
            n as i64,
            dt_a,
            a_ptr as *mut std::ffi::c_void,
            n as i64,
            dt_w,
            w_ptr as *mut std::ffi::c_void,
            dt_v,
            vl_ptr as *mut std::ffi::c_void,
            n as i64,
            dt_v,
            vr_ptr as *mut std::ffi::c_void,
            n as i64,
            compute_type,
            work_ptr as *mut std::ffi::c_void,
            wks_dev_bytes,
            host_work.as_mut_ptr() as *mut std::ffi::c_void,
            wks_host_bytes,
            info_ptr as *mut std::ffi::c_int,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_eig_f32: Xgeev failed (non-converged eigenvalues)",
            expected: vec![0],
            got: vec![info_val.unsigned_abs() as usize],
        });
    }

    // VR is stored as a column-major n×n complex matrix in cuSOLVER's
    // layout: each element occupies 2 consecutive f32s (re/im). To convert
    // to row-major while preserving the `[n, n, 2]` interleaved-complex
    // logical shape, we view VR as a column-major (2n)×n grid of f32 (each
    // complex column expands to 2 floats stacked) and transpose it to
    // row-major. After transpose, layout is `[n, 2n]`, which when reshaped
    // to `[n, n, 2]` matches our interleaved convention.
    //
    // However the more direct mental model: cuSOLVER's column-major
    // complex VR stores element (row=i, col=j) at position
    // `col_major_offset(j, i) = (j * n + i) * 2`. We want row-major at
    // `(i * n + j) * 2`. The transpose-2d kernel works on f32 strides; we
    // would need to transpose pairs of f32. Easiest is a host-side
    // permutation — pulling 2*n*n f32 down, swapping, and pushing back.
    // For typical eig sizes (n ≤ 1024) this is small.
    let host = crate::transfer::gpu_to_cpu(&d_vr, device)?;
    let mut row_major = vec![0.0_f32; 2 * n * n];
    for j in 0..n {
        for i in 0..n {
            let src = (j * n + i) * 2;
            let dst = (i * n + j) * 2;
            row_major[dst] = host[src];
            row_major[dst + 1] = host[src + 1];
        }
    }
    let d_vr_row = crate::transfer::cpu_to_gpu(&row_major, device)?;

    Ok((d_w, d_vr_row))
}

/// f64 counterpart of [`gpu_eig_f32`]. (#631)
#[cfg(feature = "cuda")]
pub fn gpu_eig_f64(
    a_dev: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>)> {
    use cudarc::cusolver::sys as csys;

    if a_dev.len() != n * n {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_eig_f64: A length mismatch",
            expected: vec![n * n],
            got: vec![a_dev.len()],
        });
    }
    if n == 0 {
        return Ok((
            crate::transfer::alloc_zeros_f64(0, device)?,
            crate::transfer::alloc_zeros_f64(0, device)?,
        ));
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;
    let params = DnParamsHandle::new()?;

    let mut d_a_col = crate::kernels::gpu_transpose_2d_f64(a_dev, n, n, device)?;
    let mut d_w = crate::transfer::alloc_zeros_f64(2 * n, device)?;
    let mut d_vr = crate::transfer::alloc_zeros_f64(2 * n * n, device)?;
    let mut d_vl_dummy = crate::transfer::alloc_zeros_f64(2, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    let novec = csys::cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR;
    let vec_mode = csys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
    let dt_a = csys::cudaDataType::CUDA_R_64F;
    let dt_w = csys::cudaDataType::CUDA_C_64F;
    let dt_v = csys::cudaDataType::CUDA_C_64F;
    let compute_type = csys::cudaDataType::CUDA_R_64F;

    let mut wks_dev_bytes: usize = 0;
    let mut wks_host_bytes: usize = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 1971 by `cusolverDnHandle::new(stream.clone())`.
    // - `params.raw()` returns the `cusolverDnParams_t` created at line 1972;
    //   the params handle outlives this call (Drop runs at function exit).
    // - `n` was validated at line 1956-1961 (A length == n*n) and is non-zero
    //   (line 1963 short-circuits the n==0 path).
    // - `a_ptr` points to an `n*n` f64 column-major buffer (`d_a_col`)
    //   produced by `gpu_transpose_2d_f64` at line 1974; `lda = n as i64`.
    //   Cast to `*const c_void` per Xgeev's type-erased ABI; `dt_a =
    //   CUDA_R_64F` (line 1982) tells cuSOLVER to interpret the bytes as f64.
    // - `w_ptr` points to a `2n` f64 buffer alloc'd at line 1976 holding
    //   complex eigenvalues as interleaved re/im pairs; `dt_w = CUDA_C_64F`
    //   matches the layout (each complex element spans 2 consecutive f64s).
    // - `vl_ptr` points to a 2-element dummy alloc'd at line 1977; cuSOLVER's
    //   X-API rejects null VL even with NOVECTOR mode (same constraint as the
    //   f32 sibling). `ldvl = n as i64`.
    // - `vr_ptr` points to a `2*n*n` f64 buffer alloc'd at line 1976 (complex
    //   eigenvectors interleaved); `ldvr = n as i64`; `dt_v = CUDA_C_64F`.
    // - `compute_type = CUDA_R_64F` (line 1985) — real f64 compute precision.
    // - `&mut wks_dev_bytes` and `&mut wks_host_bytes` are stack-resident
    //   `usize`; cuSOLVER writes the device and host workspace sizes
    //   **in bytes** per the Xgeev two-buffer ABI.
    // - This is a query-only call: it inspects dims/dtypes only, not data.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (w_ptr, _w_sync) = d_w.inner_mut().device_ptr_mut(&stream);
        let (vr_ptr, _vr_sync) = d_vr.inner_mut().device_ptr_mut(&stream);
        let (vl_ptr, _vl_sync) = d_vl_dummy.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnXgeev_bufferSize(
            dn.cu(),
            params.raw(),
            novec,
            vec_mode,
            n as i64,
            dt_a,
            a_ptr as *const std::ffi::c_void,
            n as i64,
            dt_w,
            w_ptr as *const std::ffi::c_void,
            dt_v,
            vl_ptr as *const std::ffi::c_void,
            n as i64,
            dt_v,
            vr_ptr as *const std::ffi::c_void,
            n as i64,
            compute_type,
            &mut wks_dev_bytes,
            &mut wks_host_bytes,
        )
        .result()?;
    }

    let dev_work_elems = wks_dev_bytes.div_ceil(8).max(1);
    let mut d_work = crate::transfer::alloc_zeros_f64(dev_work_elems, device)?;
    let mut host_work: Vec<u8> = vec![0u8; wks_host_bytes];

    // SAFETY:
    // - `dn.cu()`, `params.raw()` are the same valid handles from above.
    // - `a_ptr` (line 1974), `w_ptr` (line 1976), `vl_ptr` (line 1977),
    //   `vr_ptr` (line 1976) point to the same buffers proved above; Xgeev
    //   may overwrite A during Hessenberg reduction and writes complex
    //   eigenvalues into W and right eigenvectors into VR in column-major
    //   complex layout.
    // - `lda = ldvl = ldvr = n as i64` — column-major leading dims.
    // - `work_ptr` points to a device workspace of `dev_work_elems` f64
    //   elements alloc'd at line 2009; `dev_work_elems =
    //   wks_dev_bytes.div_ceil(8).max(1)` covers the queried byte budget
    //   (8-byte f64 elements, `.max(1)` guarding zero-size). cuSOLVER receives
    //   `wks_dev_bytes` so the byte count it sees is exact.
    // - `host_work.as_mut_ptr()` points to a `wks_host_bytes` `u8` buffer
    //   alloc'd at line 2010; the Vec outlives the FFI call (drops at function
    //   exit, well after `stream.synchronize()` at line 2047).
    // - `info_ptr` points to a single-element i32 alloc'd at line 1978; the
    //   status is checked at line 2048 (`read_dev_info` after synchronize at
    //   line 2047). Non-zero info ⇒ Xgeev failed.
    // - All six `_sync` guards keep the `CudaSlice` borrows alive across the
    //   FFI call.
    unsafe {
        let (a_ptr, _a_sync) = d_a_col.inner_mut().device_ptr_mut(&stream);
        let (w_ptr, _w_sync) = d_w.inner_mut().device_ptr_mut(&stream);
        let (vr_ptr, _vr_sync) = d_vr.inner_mut().device_ptr_mut(&stream);
        let (vl_ptr, _vl_sync) = d_vl_dummy.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnXgeev(
            dn.cu(),
            params.raw(),
            novec,
            vec_mode,
            n as i64,
            dt_a,
            a_ptr as *mut std::ffi::c_void,
            n as i64,
            dt_w,
            w_ptr as *mut std::ffi::c_void,
            dt_v,
            vl_ptr as *mut std::ffi::c_void,
            n as i64,
            dt_v,
            vr_ptr as *mut std::ffi::c_void,
            n as i64,
            compute_type,
            work_ptr as *mut std::ffi::c_void,
            wks_dev_bytes,
            host_work.as_mut_ptr() as *mut std::ffi::c_void,
            wks_host_bytes,
            info_ptr as *mut std::ffi::c_int,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_eig_f64: Xgeev failed",
            expected: vec![0],
            got: vec![info_val.unsigned_abs() as usize],
        });
    }

    // Same column-major complex → row-major complex permutation as f32.
    let host = crate::transfer::gpu_to_cpu(&d_vr, device)?;
    let mut row_major = vec![0.0_f64; 2 * n * n];
    for j in 0..n {
        for i in 0..n {
            let src = (j * n + i) * 2;
            let dst = (i * n + j) * 2;
            row_major[dst] = host[src];
            row_major[dst + 1] = host[src + 1];
        }
    }
    let d_vr_row = crate::transfer::cpu_to_gpu(&row_major, device)?;

    Ok((d_w, d_vr_row))
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_eig_f32(
    _a: &CudaBuffer<f32>,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    Err(GpuError::NoCudaFeature)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_eig_f64(
    _a: &CudaBuffer<f64>,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>)> {
    Err(GpuError::NoCudaFeature)
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
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
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
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (tau_ptr, _tau_sync) = d_tau.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

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
        let (a_ptr, _a_sync) = d_a.inner().device_ptr(&stream);
        let (tau_ptr, _tau_sync) = d_tau.inner().device_ptr(&stream);
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
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (tau_ptr, _tau_sync) = d_tau.inner().device_ptr(&stream);
        let (work_ptr, _work_sync) = d_work2.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info2.inner_mut().device_ptr_mut(&stream);

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

/// Compute the reduced QR decomposition of an m-by-n matrix (row-major f64).
///
/// Returns `(Q, R)` as flat row-major `Vec<f64>` with shapes:
/// - Q: [m, k]  where k = min(m, n)
/// - R: [k, n]
///
/// Uses Dgeqrf (Householder QR) followed by Dorgqr (generate Q).
#[cfg(feature = "cuda")]
pub fn gpu_qr_f64(
    data: &[f64],
    m: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(Vec<f64>, Vec<f64>)> {
    use cudarc::cusolver::sys as csys;

    if m == 0 || n == 0 {
        return Ok((vec![], vec![]));
    }

    let k = m.min(n);
    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Transpose to column-major.
    let col_major = transpose_f64(data, m, n);
    let mut d_a = upload_f64(&col_major, device)?;
    let mut d_tau = crate::transfer::alloc_zeros_f64(k, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    // Query workspace for geqrf.
    let mut lwork: i32 = 0;
    // SAFETY: dn.cu() is valid, d_a contains m*n f64 elements.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        csys::cusolverDnDgeqrf_bufferSize(
            dn.cu(),
            m as i32,
            n as i32,
            a_ptr as *mut f64,
            m as i32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f64(lwork.max(1) as usize, device)?;

    // Compute QR factorization (Householder form).
    // SAFETY: All device pointers are valid. d_a is overwritten in-place
    // with Householder reflectors (lower triangle) and R (upper triangle).
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (tau_ptr, _tau_sync) = d_tau.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _work_sync) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDgeqrf(
            dn.cu(),
            m as i32,
            n as i32,
            a_ptr as *mut f64,
            m as i32,
            tau_ptr as *mut f64,
            work_ptr as *mut f64,
            lwork,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_qr_f64: geqrf failed",
            expected: vec![0],
            got: vec![info_val as usize],
        });
    }

    // Extract R from the upper triangle of d_a (column-major).
    // R is k-by-n. We read the full m-by-n column-major buffer.
    let qr_col = download_f64(&d_a, device)?;
    let mut r_host = vec![0.0f64; k * n];
    for i in 0..k {
        for j in 0..n {
            // In column-major m-by-n: element (i, j) is at index j*m + i.
            if j >= i {
                r_host[i * n + j] = qr_col[j * m + i]; // row-major output
            }
            // else: R[i,j] = 0 (already initialized)
        }
    }

    // Generate explicit Q via Dorgqr.
    // Dorgqr overwrites d_a in-place: the first k columns become Q (m-by-k, column-major).
    let mut lwork_orgqr: i32 = 0;

    // SAFETY: dn.cu() is valid, d_a and d_tau contain valid QR factorization data.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner().device_ptr(&stream);
        let (tau_ptr, _tau_sync) = d_tau.inner().device_ptr(&stream);
        csys::cusolverDnDorgqr_bufferSize(
            dn.cu(),
            m as i32,
            k as i32,
            k as i32,
            a_ptr as *const f64,
            m as i32,
            tau_ptr as *const f64,
            &mut lwork_orgqr,
        )
        .result()?;
    }

    let mut d_work2 = crate::transfer::alloc_zeros_f64(lwork_orgqr.max(1) as usize, device)?;
    let mut d_info2 = alloc_zeros_i32(1, device)?;

    // SAFETY: d_a contains the Householder reflectors from geqrf, d_tau the
    // scalar factors. Dorgqr overwrites the first k columns of d_a with Q.
    unsafe {
        let (a_ptr, _a_sync) = d_a.inner_mut().device_ptr_mut(&stream);
        let (tau_ptr, _tau_sync) = d_tau.inner().device_ptr(&stream);
        let (work_ptr, _work_sync) = d_work2.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _info_sync) = d_info2.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDorgqr(
            dn.cu(),
            m as i32,
            k as i32,
            k as i32,
            a_ptr as *mut f64,
            m as i32,
            tau_ptr as *const f64,
            work_ptr as *mut f64,
            lwork_orgqr,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;

    let info_val2 = read_dev_info(&d_info2, device)?;
    if info_val2 != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_qr_f64: orgqr failed",
            expected: vec![0],
            got: vec![info_val2 as usize],
        });
    }

    // Download Q (m-by-k column-major from d_a, but d_a has n columns total;
    // we only need the first k columns).
    let q_full_col = download_f64(&d_a, device)?;
    let mut q_host = vec![0.0f64; m * k];
    for i in 0..m {
        for j in 0..k {
            q_host[i * k + j] = q_full_col[j * m + i]; // col-major -> row-major
        }
    }

    Ok((q_host, r_host))
}

// ===========================================================================
// Symmetric / Hermitian eigendecomposition (eigh / eigvalsh)
//
// `cusolverDn{S,D}syevd` computes eigenvalues + eigenvectors of a real
// symmetric matrix. The routine works in-place on `A` (overwriting it
// with the eigenvectors) and writes eigenvalues into a separate buffer.
//
// Memory layout note: cuSOLVER expects column-major. For a real
// **symmetric** matrix the row-major and column-major layouts are
// byte-identical (transpose of a symmetric matrix is itself), so we can
// pass a row-major buffer to cuSOLVER without an explicit transpose.
// The OUTPUT eigenvector matrix is general (non-symmetric), so we
// transpose it back to row-major using `gpu_transpose_2d_*` —
// fully on-device.
//
// /rust-gpu-discipline: every step here either runs on GPU memory (cuSOLVER
// / strided_copy / memcpy_dtod) or is metadata bookkeeping. No host
// bounce of the matrix data.
// ===========================================================================

/// Eigendecomposition of an `n × n` real symmetric matrix (f32).
///
/// Takes a GPU-resident row-major buffer and returns
/// `(eigenvalues_ascending, eigenvectors_row_major)` both on-device. The
/// caller's input buffer is **not** mutated — we clone it before the
/// in-place cuSOLVER call.
///
/// Eigenvalues are sorted ascending. Eigenvectors are returned with
/// column `j` of the result tensor equal to the `j`-th eigenvector,
/// matching the row-major convention used by `ferrotorch_core::linalg::eigh`.
#[cfg(feature = "cuda")]
pub fn gpu_eigh_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        // Empty matrix — return empty eigenvalue + zero-sized eigenvector
        // buffers rather than calling cuSOLVER.
        return Ok((
            crate::transfer::alloc_zeros_f32(0, device)?,
            crate::transfer::alloc_zeros_f32(0, device)?,
        ));
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Clone input on GPU (cuSOLVER syevd writes in-place over A).
    let mut d_a = crate::transfer::alloc_zeros_f32(n * n, device)?;
    stream.memcpy_dtod(input.inner(), d_a.inner_mut())?;

    // Output eigenvalue buffer + devInfo.
    let mut d_w = crate::transfer::alloc_zeros_f32(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    // Workspace size query.
    let jobz = csys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
    let uplo = csys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created by `cusolverDnHandle::new(stream.clone())` earlier in this
    //   function (just before the `d_a` clone). The handle outlives this
    //   call (dn drops at function exit).
    // - `n` is non-zero (the n==0 path short-circuits earlier with empty
    //   buffers); the cuSOLVER docs require `lda >= max(1, n)`, satisfied by
    //   `lda = n as i32`.
    // - The device pointers from `d_a.inner().device_ptr(&stream).0` and
    //   `d_w.inner().device_ptr(&stream).0` are extracted from the freshly-
    //   allocated buffers (line 2895 area: `d_a` is `n*n` f32, `d_w` is `n`
    //   f32 from `alloc_zeros_f32`); these are read-only in the buffer-size
    //   query (no compute, no write).
    // - For real symmetric matrices the row-major and column-major byte
    //   layouts coincide (transpose of a symmetric matrix is itself), per
    //   the comment at lines ~2401-2407 — so passing the row-major
    //   `d_a = memcpy_dtod(input)` directly is safe.
    // - `uplo = CUBLAS_FILL_MODE_UPPER`: cuSOLVER reads only the upper
    //   triangle of A (which equals the lower for symmetric inputs).
    // - `jobz = CUSOLVER_EIG_MODE_VECTOR` requests both eigenvalues and
    //   eigenvectors, so cuSOLVER will need the larger of the two workspace
    //   sizes — that's what `&mut lwork` will receive.
    // - `&mut lwork` is stack-resident `i32`; cuSOLVER writes the workspace
    //   size in **f32 elements** (not bytes) per the syevd_bufferSize ABI.
    // - The temporary `device_ptr` borrows held by `.0` extraction live only
    //   for the duration of the call expression (cudarc's `DevicePtr`
    //   guard keeps the underlying allocation alive while in scope).
    unsafe {
        csys::cusolverDnSsyevd_bufferSize(
            dn.cu(),
            jobz,
            uplo,
            n as i32,
            d_a.inner().device_ptr(&stream).0 as *const f32,
            n as i32,
            d_w.inner().device_ptr(&stream).0 as *const f32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f32(lwork.max(1) as usize, device)?;

    // Run syevd.
    // SAFETY:
    // - `dn.cu()` is the same valid handle proved above.
    // - `a_ptr` points to the `n*n` f32 buffer (`d_a`); cuSOLVER overwrites
    //   it in place: with `jobz = VECTOR` the eigenvectors are written into
    //   A column-major (each column a normalized eigenvector).
    // - `lda = n as i32` matches the leading dimension.
    // - `uplo = CUBLAS_FILL_MODE_UPPER` matches the buffer-size query.
    // - `w_ptr` points to the `n`-element f32 buffer `d_w`; cuSOLVER writes
    //   eigenvalues here in ascending order.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f32 elements
    //   allocated immediately above from the buffer-size query; cuSOLVER
    //   receives the queried `lwork`, so the buffer is at least as large as
    //   it claimed to need. `.max(1)` prevents zero-sized alloc.
    // - `info_ptr` points to a single-element i32; the status is checked at
    //   line 2946 (`read_dev_info` after synchronize at line 2945). A
    //   positive info_val indicates non-converged eigenvalues; converted to
    //   `GpuError::ShapeMismatch` at line 2948.
    // - All four `_` placeholder bindings receive the `_sync` guards from
    //   `device_ptr_mut`; they remain in scope (until end of unsafe block)
    //   to keep the `CudaSlice` borrows live across the FFI call. The
    //   bare-`_` placeholder is allowed because the guards are still bound
    //   to a tuple slot for the block's duration.
    unsafe {
        let (a_ptr, _) = d_a.inner_mut().device_ptr_mut(&stream);
        let (w_ptr, _) = d_w.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnSsyevd(
            dn.cu(),
            jobz,
            uplo,
            n as i32,
            a_ptr as *mut f32,
            n as i32,
            w_ptr as *mut f32,
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
            op: "gpu_eigh_f32",
            expected: vec![0],
            got: vec![info_val.unsigned_abs() as usize],
        });
    }

    // d_a now holds the eigenvectors in column-major (each column is an
    // eigenvector). Transpose back to row-major so column `j` of the
    // returned tensor (in row-major) is the `j`-th eigenvector.
    let v_rm = crate::kernels::gpu_transpose_2d(&d_a, n, n, device)?;

    Ok((d_w, v_rm))
}

/// f64 variant of [`gpu_eigh_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_eigh_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<f64>)> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        return Ok((
            crate::transfer::alloc_zeros_f64(0, device)?,
            crate::transfer::alloc_zeros_f64(0, device)?,
        ));
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    let mut d_a = crate::transfer::alloc_zeros_f64(n * n, device)?;
    stream.memcpy_dtod(input.inner(), d_a.inner_mut())?;

    let mut d_w = crate::transfer::alloc_zeros_f64(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    let jobz = csys::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR;
    let uplo = csys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created earlier in this function (just before the `d_a` clone).
    //   The handle outlives this call (dn drops at function exit).
    // - `n` is non-zero (n==0 path short-circuits earlier with empty
    //   buffers); `lda = n as i32` satisfies `lda >= max(1, n)`.
    // - The device pointers from `d_a.inner().device_ptr(&stream).0` and
    //   `d_w.inner().device_ptr(&stream).0` come from the freshly-allocated
    //   `d_a` (`n*n` f64 from `alloc_zeros_f64`, populated via memcpy_dtod
    //   from the input) and `d_w` (`n` f64 zero-init). Read-only here.
    // - For real symmetric matrices the row-major and column-major byte
    //   layouts coincide — the row-major `d_a` clone is a valid input.
    // - `uplo = CUBLAS_FILL_MODE_UPPER`: cuSOLVER reads only the upper
    //   triangle (equals lower for symmetric).
    // - `jobz = CUSOLVER_EIG_MODE_VECTOR` requests both eigenvalues and
    //   eigenvectors; cuSOLVER will report the larger workspace size.
    // - `&mut lwork` is stack-resident `i32`; cuSOLVER writes the f64
    //   workspace size in **elements** per the syevd_bufferSize ABI.
    unsafe {
        csys::cusolverDnDsyevd_bufferSize(
            dn.cu(),
            jobz,
            uplo,
            n as i32,
            d_a.inner().device_ptr(&stream).0 as *const f64,
            n as i32,
            d_w.inner().device_ptr(&stream).0 as *const f64,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f64(lwork.max(1) as usize, device)?;

    // SAFETY:
    // - `dn.cu()` is the same valid handle proved above.
    // - `a_ptr` points to the `n*n` f64 buffer (`d_a`); cuSOLVER overwrites
    //   it in place — with `jobz = VECTOR` the eigenvectors are written
    //   into A column-major (each column a normalized eigenvector).
    // - `lda = n as i32` matches the leading dimension.
    // - `uplo = CUBLAS_FILL_MODE_UPPER` matches the buffer-size query.
    // - `w_ptr` points to the `n`-element f64 buffer `d_w`; cuSOLVER writes
    //   eigenvalues here in ascending order.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f64 elements
    //   allocated immediately above from the buffer-size query; cuSOLVER
    //   receives the queried `lwork`, so the buffer is at least as large
    //   as it claimed to need. `.max(1)` prevents zero-sized alloc.
    // - `info_ptr` points to a single-element i32; status checked at line
    //   3029 (`read_dev_info` after synchronize at line 3028). Positive
    //   info_val ⇒ non-converged eigenvalues, converted to `ShapeMismatch`.
    // - The four `_` placeholder bindings still hold the `_sync` guards
    //   for the unsafe block's duration, keeping `CudaSlice` borrows live
    //   across the FFI call.
    unsafe {
        let (a_ptr, _) = d_a.inner_mut().device_ptr_mut(&stream);
        let (w_ptr, _) = d_w.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDsyevd(
            dn.cu(),
            jobz,
            uplo,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            w_ptr as *mut f64,
            work_ptr as *mut f64,
            lwork,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_eigh_f64",
            expected: vec![0],
            got: vec![info_val.unsigned_abs() as usize],
        });
    }

    let v_rm = crate::kernels::gpu_transpose_2d_f64(&d_a, n, n, device)?;
    Ok((d_w, v_rm))
}

/// Eigenvalues only of an `n × n` real symmetric matrix (f32).
///
/// Same as [`gpu_eigh_f32`] but skips the eigenvector computation —
/// faster, and avoids the row/column-major transpose at the end.
#[cfg(feature = "cuda")]
pub fn gpu_eigvalsh_f32(
    input: &CudaBuffer<f32>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        return crate::transfer::alloc_zeros_f32(0, device);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    // Even with NoVector, syevd writes garbage to A — clone to protect input.
    let mut d_a = crate::transfer::alloc_zeros_f32(n * n, device)?;
    stream.memcpy_dtod(input.inner(), d_a.inner_mut())?;

    let mut d_w = crate::transfer::alloc_zeros_f32(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    let jobz = csys::cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR;
    let uplo = csys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 3145 by `cusolverDnHandle::new(stream.clone())`. The
    //   handle outlives this call (dn drops at function exit).
    // - `n` is non-zero (line 3140 short-circuits the n==0 path); `lda =
    //   n as i32` satisfies cuSOLVER's `lda >= max(1, n)` requirement.
    // - The pointers from `d_a.inner().device_ptr(&stream).0` and
    //   `d_w.inner().device_ptr(&stream).0` come from the freshly-allocated
    //   `d_a` (`n*n` f32, line 3148; populated via `memcpy_dtod` at line 3149)
    //   and `d_w` (`n` f32 zero-init, line 3151). Read-only in this query.
    // - For real symmetric matrices the row-major and column-major layouts
    //   coincide (per the comment at lines ~2401-2407); the row-major input
    //   clone is a valid input.
    // - `uplo = CUBLAS_FILL_MODE_UPPER`: cuSOLVER reads only the upper
    //   triangle (equals lower for symmetric).
    // - `jobz = CUSOLVER_EIG_MODE_NOVECTOR` requests eigenvalues only —
    //   cuSOLVER skips eigenvector accumulation and reports the smaller
    //   workspace size for this mode.
    // - `&mut lwork` is stack-resident `i32`; cuSOLVER writes the workspace
    //   size in **f32 elements** per the syevd_bufferSize ABI.
    unsafe {
        csys::cusolverDnSsyevd_bufferSize(
            dn.cu(),
            jobz,
            uplo,
            n as i32,
            d_a.inner().device_ptr(&stream).0 as *const f32,
            n as i32,
            d_w.inner().device_ptr(&stream).0 as *const f32,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f32(lwork.max(1) as usize, device)?;

    // SAFETY:
    // - `dn.cu()` is the same valid handle proved above.
    // - `a_ptr` points to the `n*n` f32 buffer `d_a` (line 3148). With
    //   `jobz = NOVECTOR`, cuSOLVER still overwrites A with intermediate
    //   tridiagonal data ("syevd writes garbage to A — clone to protect
    //   input" per the comment at line 3147), so we must use a cloned A and
    //   the input remains untouched.
    // - `lda = n as i32` matches the leading dimension.
    // - `uplo = CUBLAS_FILL_MODE_UPPER` matches the buffer-size query.
    // - `w_ptr` points to the `n`-element f32 buffer `d_w`; cuSOLVER writes
    //   eigenvalues here in ascending order.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f32 elements
    //   allocated immediately above from the buffer-size query; cuSOLVER
    //   receives the queried `lwork`. `.max(1)` prevents zero-sized alloc.
    // - `info_ptr` points to a single-element i32; status checked at line
    //   3195 (`read_dev_info` after synchronize at line 3194). Positive
    //   info ⇒ non-converged eigenvalues; converted to `ShapeMismatch`.
    // - The four `_` placeholder bindings hold `_sync` guards for the
    //   block's duration, keeping `CudaSlice` borrows live across the FFI
    //   call.
    unsafe {
        let (a_ptr, _) = d_a.inner_mut().device_ptr_mut(&stream);
        let (w_ptr, _) = d_w.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnSsyevd(
            dn.cu(),
            jobz,
            uplo,
            n as i32,
            a_ptr as *mut f32,
            n as i32,
            w_ptr as *mut f32,
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
            op: "gpu_eigvalsh_f32",
            expected: vec![0],
            got: vec![info_val.unsigned_abs() as usize],
        });
    }

    Ok(d_w)
}

/// f64 variant of [`gpu_eigvalsh_f32`].
#[cfg(feature = "cuda")]
pub fn gpu_eigvalsh_f64(
    input: &CudaBuffer<f64>,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    use cudarc::cusolver::sys as csys;

    if n == 0 {
        return crate::transfer::alloc_zeros_f64(0, device);
    }

    let stream = device.stream();
    let dn = cusolver_safe::DnHandle::new(stream.clone())?;

    let mut d_a = crate::transfer::alloc_zeros_f64(n * n, device)?;
    stream.memcpy_dtod(input.inner(), d_a.inner_mut())?;

    let mut d_w = crate::transfer::alloc_zeros_f64(n, device)?;
    let mut d_info = alloc_zeros_i32(1, device)?;

    let jobz = csys::cusolverEigMode_t::CUSOLVER_EIG_MODE_NOVECTOR;
    let uplo = csys::cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
    let mut lwork: i32 = 0;
    // SAFETY:
    // - `dn.cu()` returns a valid `cusolverDnHandle_t` bound to `stream`,
    //   created at line 3221 by `cusolverDnHandle::new(stream.clone())`. The
    //   handle outlives this call (dn drops at function exit).
    // - `n` is non-zero (line 3216 short-circuits the n==0 path); `lda =
    //   n as i32` satisfies cuSOLVER's `lda >= max(1, n)` requirement.
    // - The pointers from `d_a.inner().device_ptr(&stream).0` and
    //   `d_w.inner().device_ptr(&stream).0` come from the freshly-allocated
    //   `d_a` (`n*n` f64, line 3223; populated via `memcpy_dtod` at line 3224)
    //   and `d_w` (`n` f64 zero-init, line 3226). Read-only in this query.
    // - For real symmetric matrices the row-major and column-major layouts
    //   coincide; the row-major input clone is a valid input.
    // - `uplo = CUBLAS_FILL_MODE_UPPER`: cuSOLVER reads only the upper
    //   triangle.
    // - `jobz = CUSOLVER_EIG_MODE_NOVECTOR` requests eigenvalues only.
    // - `&mut lwork` is stack-resident `i32`; cuSOLVER writes the f64
    //   workspace size in **elements** per the syevd_bufferSize ABI.
    unsafe {
        csys::cusolverDnDsyevd_bufferSize(
            dn.cu(),
            jobz,
            uplo,
            n as i32,
            d_a.inner().device_ptr(&stream).0 as *const f64,
            n as i32,
            d_w.inner().device_ptr(&stream).0 as *const f64,
            &mut lwork,
        )
        .result()?;
    }

    let mut d_work = crate::transfer::alloc_zeros_f64(lwork.max(1) as usize, device)?;

    // SAFETY:
    // - `dn.cu()` is the same valid handle proved above.
    // - `a_ptr` points to the `n*n` f64 buffer `d_a` (line 3223). With
    //   `jobz = NOVECTOR`, cuSOLVER still overwrites A with intermediate
    //   tridiagonal data, hence the `memcpy_dtod` clone at line 3224 to
    //   protect the caller's input.
    // - `lda = n as i32` matches the leading dimension.
    // - `uplo = CUBLAS_FILL_MODE_UPPER` matches the buffer-size query.
    // - `w_ptr` points to the `n`-element f64 buffer `d_w`; cuSOLVER writes
    //   eigenvalues here in ascending order.
    // - `work_ptr` points to a workspace of `lwork.max(1)` f64 elements
    //   allocated immediately above from the buffer-size query; cuSOLVER
    //   receives the queried `lwork`, so the buffer is at least as large
    //   as it claimed to need. `.max(1)` prevents zero-sized alloc.
    // - `info_ptr` points to a single-element i32; status checked at line
    //   3270 (`read_dev_info` after synchronize at line 3269). Positive
    //   info_val ⇒ non-converged eigenvalues; converted to `ShapeMismatch`.
    // - The four `_` placeholder bindings hold `_sync` guards for the
    //   block's duration, keeping `CudaSlice` borrows live across the FFI
    //   call.
    unsafe {
        let (a_ptr, _) = d_a.inner_mut().device_ptr_mut(&stream);
        let (w_ptr, _) = d_w.inner_mut().device_ptr_mut(&stream);
        let (work_ptr, _) = d_work.inner_mut().device_ptr_mut(&stream);
        let (info_ptr, _) = d_info.inner_mut().device_ptr_mut(&stream);

        csys::cusolverDnDsyevd(
            dn.cu(),
            jobz,
            uplo,
            n as i32,
            a_ptr as *mut f64,
            n as i32,
            w_ptr as *mut f64,
            work_ptr as *mut f64,
            lwork,
            info_ptr as *mut i32,
        )
        .result()?;
    }

    stream.synchronize()?;
    let info_val = read_dev_info(&d_info, device)?;
    if info_val != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_eigvalsh_f64",
            expected: vec![0],
            got: vec![info_val.unsigned_abs() as usize],
        });
    }

    Ok(d_w)
}

// ---------------------------------------------------------------------------
// Stubs — always return [`GpuError::NoCudaFeature`] when `cuda` is disabled.
// ---------------------------------------------------------------------------

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_svd_f32(
    _data: &[f32],
    _m: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_svd_f64(
    _data: &[f64],
    _m: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_cholesky_f32(_data: &[f32], _n: usize, _device: &GpuDevice) -> GpuResult<Vec<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_cholesky_f64(_data: &[f64], _n: usize, _device: &GpuDevice) -> GpuResult<Vec<f64>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_solve_f32(
    _a_data: &[f32],
    _b_data: &[f32],
    _n: usize,
    _nrhs: usize,
    _device: &GpuDevice,
) -> GpuResult<Vec<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_solve_f64(
    _a_data: &[f64],
    _b_data: &[f64],
    _n: usize,
    _nrhs: usize,
    _device: &GpuDevice,
) -> GpuResult<Vec<f64>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_lu_factor_f32(
    _a_dev: &CudaBuffer<f32>,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<i32>)> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_lu_factor_f64(
    _a_dev: &CudaBuffer<f64>,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f64>, CudaBuffer<i32>)> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_qr_f32(
    _data: &[f32],
    _m: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<(Vec<f32>, Vec<f32>)> {
    Err(GpuError::NoCudaFeature)
}

/// Stub — always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_qr_f64(
    _data: &[f64],
    _m: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<(Vec<f64>, Vec<f64>)> {
    Err(GpuError::NoCudaFeature)
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
        let mut llt = [0.0f32; 9];
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
        let mut ax = [0.0f32; 4];
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
        let mut recon = [0.0f32; 9];
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
