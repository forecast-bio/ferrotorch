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

/// Elementwise `out = a / b`.
#[cube(launch_unchecked)]
pub fn kernel_div<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] / b[ABSOLUTE_POS];
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

/// Elementwise `out = -x`.
#[cube(launch_unchecked)]
pub fn kernel_neg<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::new(0.0) - x[ABSOLUTE_POS];
    }
}

/// Elementwise `out = |x|`.
#[cube(launch_unchecked)]
pub fn kernel_abs<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::abs(x[ABSOLUTE_POS]);
    }
}

/// Elementwise `out = exp(x)`.
#[cube(launch_unchecked)]
pub fn kernel_exp<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::exp(x[ABSOLUTE_POS]);
    }
}

/// Elementwise `out = ln(x)` (natural log).
#[cube(launch_unchecked)]
pub fn kernel_ln<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::ln(x[ABSOLUTE_POS]);
    }
}

/// Elementwise `out = sqrt(x)`.
#[cube(launch_unchecked)]
pub fn kernel_sqrt<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::sqrt(x[ABSOLUTE_POS]);
    }
}

/// Elementwise `out = sin(x)`.
#[cube(launch_unchecked)]
pub fn kernel_sin<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::sin(x[ABSOLUTE_POS]);
    }
}

/// Elementwise `out = cos(x)`.
#[cube(launch_unchecked)]
pub fn kernel_cos<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::cos(x[ABSOLUTE_POS]);
    }
}

/// Elementwise `out = tanh(x)`.
#[cube(launch_unchecked)]
pub fn kernel_tanh<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = F::tanh(x[ABSOLUTE_POS]);
    }
}

/// Elementwise `out = 1 / (1 + exp(-x))` — logistic sigmoid.
#[cube(launch_unchecked)]
pub fn kernel_sigmoid<F: Float>(x: &Array<F>, out: &mut Array<F>) {
    if ABSOLUTE_POS < out.len() {
        let neg_x = F::new(0.0) - x[ABSOLUTE_POS];
        out[ABSOLUTE_POS] = F::new(1.0) / (F::new(1.0) + F::exp(neg_x));
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

// ---------------------------------------------------------------------------
// Unary + binary helpers shared by every elementwise op
// ---------------------------------------------------------------------------

/// Upload `x`, launch `launcher`, read back the result.
fn run_unary<R, L>(client: &ComputeClient<R>, x: &[f32], launcher: L) -> Vec<f32>
where
    R: Runtime,
    L: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>),
{
    let n = x.len();
    let size_bytes = n * std::mem::size_of::<f32>();

    let x_handle = client.create_from_slice(f32::as_bytes(x));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = elementwise_launch_dims(n as u32);
    let in_arg = unsafe { ArrayArg::from_raw_parts::<f32>(&x_handle, n, 1) };
    let out_arg = unsafe { ArrayArg::from_raw_parts::<f32>(&out_handle, n, 1) };
    launcher(client, count, dim, in_arg, out_arg);

    let bytes = client.read_one(out_handle);
    f32::from_bytes(&bytes)[..n].to_vec()
}

/// Upload `a` and `b`, launch `launcher`, read back the result.
fn run_binary<R, L>(client: &ComputeClient<R>, a: &[f32], b: &[f32], launcher: L) -> Vec<f32>
where
    R: Runtime,
    L: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>, ArrayArg<R>),
{
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let size_bytes = n * std::mem::size_of::<f32>();

    let a_handle = client.create_from_slice(f32::as_bytes(a));
    let b_handle = client.create_from_slice(f32::as_bytes(b));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = elementwise_launch_dims(n as u32);
    let a_arg = unsafe { ArrayArg::from_raw_parts::<f32>(&a_handle, n, 1) };
    let b_arg = unsafe { ArrayArg::from_raw_parts::<f32>(&b_handle, n, 1) };
    let out_arg = unsafe { ArrayArg::from_raw_parts::<f32>(&out_handle, n, 1) };
    launcher(client, count, dim, a_arg, b_arg, out_arg);

    let bytes = client.read_one(out_handle);
    f32::from_bytes(&bytes)[..n].to_vec()
}

// Per-kernel `run_*` helpers: each one is just a thin wrapper around
// `run_unary` / `run_binary` that plugs in the generated
// `kernel_*::launch_unchecked` symbol. Two macros stamp them out.

macro_rules! define_unary_runner {
    ($run_fn:ident, $kernel:ident) => {
        #[doc = concat!("Upload `x`, run `", stringify!($kernel), "`, read back the result.")]
        pub fn $run_fn<R: Runtime>(client: &ComputeClient<R>, x: &[f32]) -> Vec<f32> {
            run_unary::<R, _>(client, x, |client, count, dim, input, output| unsafe {
                $kernel::launch_unchecked::<f32, R>(client, count, dim, input, output)
                    .expect(concat!("cubecl ", stringify!($kernel), " launch failed"));
            })
        }
    };
}

macro_rules! define_binary_runner {
    ($run_fn:ident, $kernel:ident) => {
        #[doc = concat!("Upload `a` and `b`, run `", stringify!($kernel), "`, read back the result.")]
        pub fn $run_fn<R: Runtime>(
            client: &ComputeClient<R>,
            a: &[f32],
            b: &[f32],
        ) -> Vec<f32> {
            run_binary::<R, _>(client, a, b, |client, count, dim, a, b, out| unsafe {
                $kernel::launch_unchecked::<f32, R>(client, count, dim, a, b, out)
                    .expect(concat!("cubecl ", stringify!($kernel), " launch failed"));
            })
        }
    };
}

// Binary ops
define_binary_runner!(run_add, kernel_add);
define_binary_runner!(run_sub, kernel_sub);
define_binary_runner!(run_mul, kernel_mul);
define_binary_runner!(run_div, kernel_div);

// Unary ops
define_unary_runner!(run_relu, kernel_relu);
define_unary_runner!(run_neg, kernel_neg);
define_unary_runner!(run_abs, kernel_abs);
define_unary_runner!(run_exp, kernel_exp);
define_unary_runner!(run_ln, kernel_ln);
define_unary_runner!(run_sqrt, kernel_sqrt);
define_unary_runner!(run_sin, kernel_sin);
define_unary_runner!(run_cos, kernel_cos);
define_unary_runner!(run_tanh, kernel_tanh);
define_unary_runner!(run_sigmoid, kernel_sigmoid);

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
