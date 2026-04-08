//! CubeCL kernel definitions used by `ferrotorch-cubecl`.
//!
//! Each kernel is written with the `#[cube]` macro, so the same source
//! compiles to CUDA PTX, AMD HIP, and WGPU/Metal/Vulkan depending on which
//! runtime it is launched with.
//!
//! The public helpers in this module take a `ComputeClient<R>` and host
//! slices, upload the inputs, dispatch the kernel, and read the result back
//! to the host. They are intentionally runtime-generic so that `ops.rs` can
//! match on the enum of available clients (Wgpu / Cuda / Rocm) and call the
//! same helper for each branch.

use cubecl::prelude::*;

// ---------------------------------------------------------------------------
// Kernel definitions
// ---------------------------------------------------------------------------

/// Elementwise `out = a + b` — one element per unit.
#[cube(launch_unchecked)]
pub fn kernel_add<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

/// Elementwise `out = a - b`.
#[cube(launch_unchecked)]
pub fn kernel_sub<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] - b[ABSOLUTE_POS];
    }
}

/// Elementwise `out = a * b`.
#[cube(launch_unchecked)]
pub fn kernel_mul<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] * b[ABSOLUTE_POS];
    }
}

/// Elementwise `out = max(x, 0)`.
#[cube(launch_unchecked)]
pub fn kernel_relu<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        let v = x[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = F::max(v, F::new(0.0));
    }
}

/// Naive row-major matmul: `out[m, n] = sum_k a[m, k] * b[k, n]`.
///
/// One cube-unit computes one output element. The `m`, `k`, and `n` scalars
/// are passed at launch time via `ScalarArg` so the same compiled kernel
/// handles arbitrary shapes. They are `usize` to match `ABSOLUTE_POS` and
/// `Array::len()` which are both `usize` inside a cube kernel.
#[cube(launch_unchecked)]
pub fn kernel_matmul_naive<F: Float>(
    a: &Array<F>,
    b: &Array<F>,
    out: &mut Array<F>,
    m: u32,
    k: u32,
    n: u32,
) {
    let m_u = m as usize;
    let k_u = k as usize;
    let n_u = n as usize;
    let total = m_u * n_u;
    if ABSOLUTE_POS < total {
        let row = ABSOLUTE_POS / n_u;
        let col = ABSOLUTE_POS % n_u;
        let mut acc = F::new(0.0);
        for i in 0..k_u {
            acc += a[row * k_u + i] * b[i * n_u + col];
        }
        out[ABSOLUTE_POS] = acc;
    }
}

// ---------------------------------------------------------------------------
// Launch helpers — runtime-generic, f32-concrete
// ---------------------------------------------------------------------------

/// Choose a 1-D cube dim and cube count that cover `n` elements when each
/// unit processes exactly one element.
fn elementwise_launch_dims(n: u32) -> (CubeCount, CubeDim) {
    // 256 units per cube is a safe default across all backends; wgpu requires
    // the workgroup size to be compiled into the shader, but CubeCL handles
    // that through `CubeDim`.
    let units_per_cube: u32 = 256;
    let num_cubes = n.div_ceil(units_per_cube).max(1);
    (
        CubeCount::Static(num_cubes, 1, 1),
        CubeDim::new_1d(units_per_cube),
    )
}

/// Upload `a` and `b`, run `kernel_add`, read back the result.
pub fn run_add<R: Runtime>(client: &ComputeClient<R>, a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let size_bytes = n * std::mem::size_of::<f32>();

    let a_handle = client.create_from_slice(f32::as_bytes(a));
    let b_handle = client.create_from_slice(f32::as_bytes(b));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = elementwise_launch_dims(n as u32);
    unsafe {
        kernel_add::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&b_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, n, 1),
        )
        .expect("cubecl add kernel launch failed");
    }

    let bytes = client.read_one(out_handle);
    f32::from_bytes(&bytes)[..n].to_vec()
}

/// Upload `a` and `b`, run `kernel_sub`, read back the result.
pub fn run_sub<R: Runtime>(client: &ComputeClient<R>, a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let size_bytes = n * std::mem::size_of::<f32>();

    let a_handle = client.create_from_slice(f32::as_bytes(a));
    let b_handle = client.create_from_slice(f32::as_bytes(b));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = elementwise_launch_dims(n as u32);
    unsafe {
        kernel_sub::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&b_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, n, 1),
        )
        .expect("cubecl sub kernel launch failed");
    }

    let bytes = client.read_one(out_handle);
    f32::from_bytes(&bytes)[..n].to_vec()
}

/// Upload `a` and `b`, run `kernel_mul`, read back the result.
pub fn run_mul<R: Runtime>(client: &ComputeClient<R>, a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let size_bytes = n * std::mem::size_of::<f32>();

    let a_handle = client.create_from_slice(f32::as_bytes(a));
    let b_handle = client.create_from_slice(f32::as_bytes(b));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = elementwise_launch_dims(n as u32);
    unsafe {
        kernel_mul::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts::<f32>(&a_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&b_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, n, 1),
        )
        .expect("cubecl mul kernel launch failed");
    }

    let bytes = client.read_one(out_handle);
    f32::from_bytes(&bytes)[..n].to_vec()
}

/// Upload `x`, run `kernel_relu`, read back the result.
pub fn run_relu<R: Runtime>(client: &ComputeClient<R>, x: &[f32]) -> Vec<f32> {
    let n = x.len();
    let size_bytes = n * std::mem::size_of::<f32>();

    let x_handle = client.create_from_slice(f32::as_bytes(x));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = elementwise_launch_dims(n as u32);
    unsafe {
        kernel_relu::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts::<f32>(&x_handle, n, 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, n, 1),
        )
        .expect("cubecl relu kernel launch failed");
    }

    let bytes = client.read_one(out_handle);
    f32::from_bytes(&bytes)[..n].to_vec()
}

/// Upload `a` and `b`, run `kernel_matmul_naive`, read back the result.
///
/// `a` is `[m * k]` row-major, `b` is `[k * n]` row-major, output is
/// `[m * n]` row-major. The caller is responsible for verifying sizes.
pub fn run_matmul<R: Runtime>(
    client: &ComputeClient<R>,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    let out_len = m * n;
    let size_bytes = out_len * std::mem::size_of::<f32>();

    let a_handle = client.create_from_slice(f32::as_bytes(a));
    let b_handle = client.create_from_slice(f32::as_bytes(b));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = elementwise_launch_dims(out_len as u32);
    unsafe {
        kernel_matmul_naive::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts::<f32>(&a_handle, a.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&b_handle, b.len(), 1),
            ArrayArg::from_raw_parts::<f32>(&out_handle, out_len, 1),
            ScalarArg::new(m as u32),
            ScalarArg::new(k as u32),
            ScalarArg::new(n as u32),
        )
        .expect("cubecl matmul kernel launch failed");
    }

    let bytes = client.read_one(out_handle);
    f32::from_bytes(&bytes)[..out_len].to_vec()
}
