//! CubeCL kernel definitions used by `ferrotorch-cubecl`.
//!
//! Each kernel is written with the `#[cube]` macro, so the same source
//! compiles to CUDA PTX, AMD HIP, and WGPU/Metal/Vulkan depending on which
//! runtime it is launched with.
//!
//! The public helpers in this module take a `ComputeClient<R>` and host
//! slices, upload the inputs, and dispatch the kernel. They return a
//! `(cubecl::server::Handle, usize)` pair — the on-device result buffer plus
//! its element count — **without reading back to the host**. Callers that
//! need CPU-resident data are responsible for calling
//! `client.read_one(handle)` themselves. This mirrors the idiom established
//! by `quant.rs` and `grammar.rs`: keep data device-resident and let the
//! boundary crate decide when to read back. ADR #663 item 4.

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

// ---------------------------------------------------------------------------
// Orthogonal polynomial kernels — three-term recurrences. (#577)
//
// Each thread handles one input element, runs the polynomial recurrence up
// to degree `n`, and writes the final value. The CPU evaluators in
// `ferrotorch-core::special` use the identical recurrences in f64; here we
// stay in `F` (f32 today) so the result lives entirely on device.
// ---------------------------------------------------------------------------

/// Chebyshev T_n(x): T_0=1, T_1=x, T_{k+1} = 2x T_k - T_{k-1}.
#[cube(launch_unchecked)]
pub fn kernel_chebyshev_t<F: Float>(x: &Array<F>, out: &mut Array<F>, n: u32) {
    if ABSOLUTE_POS < out.len() {
        let xv = x[ABSOLUTE_POS];
        let two_x = F::new(2.0) * xv;
        let n_u = n as usize;
        let mut prev2 = F::new(1.0);
        let mut prev1 = xv;
        if n_u == 0 {
            out[ABSOLUTE_POS] = prev2;
        } else if n_u == 1 {
            out[ABSOLUTE_POS] = prev1;
        } else {
            for _ in 2..=n_u {
                let next = two_x * prev1 - prev2;
                prev2 = prev1;
                prev1 = next;
            }
            out[ABSOLUTE_POS] = prev1;
        }
    }
}

/// Chebyshev U_n(x): U_0=1, U_1=2x, U_{k+1} = 2x U_k - U_{k-1}.
#[cube(launch_unchecked)]
pub fn kernel_chebyshev_u<F: Float>(x: &Array<F>, out: &mut Array<F>, n: u32) {
    if ABSOLUTE_POS < out.len() {
        let xv = x[ABSOLUTE_POS];
        let two_x = F::new(2.0) * xv;
        let n_u = n as usize;
        let mut prev2 = F::new(1.0);
        let mut prev1 = two_x;
        if n_u == 0 {
            out[ABSOLUTE_POS] = prev2;
        } else if n_u == 1 {
            out[ABSOLUTE_POS] = prev1;
        } else {
            for _ in 2..=n_u {
                let next = two_x * prev1 - prev2;
                prev2 = prev1;
                prev1 = next;
            }
            out[ABSOLUTE_POS] = prev1;
        }
    }
}

/// Chebyshev V_n(x): V_0=1, V_1=2x-1, V_{k+1} = 2x V_k - V_{k-1}.
#[cube(launch_unchecked)]
pub fn kernel_chebyshev_v<F: Float>(x: &Array<F>, out: &mut Array<F>, n: u32) {
    if ABSOLUTE_POS < out.len() {
        let xv = x[ABSOLUTE_POS];
        let two_x = F::new(2.0) * xv;
        let n_u = n as usize;
        let mut prev2 = F::new(1.0);
        let mut prev1 = two_x - F::new(1.0);
        if n_u == 0 {
            out[ABSOLUTE_POS] = prev2;
        } else if n_u == 1 {
            out[ABSOLUTE_POS] = prev1;
        } else {
            for _ in 2..=n_u {
                let next = two_x * prev1 - prev2;
                prev2 = prev1;
                prev1 = next;
            }
            out[ABSOLUTE_POS] = prev1;
        }
    }
}

/// Chebyshev W_n(x): W_0=1, W_1=2x+1, W_{k+1} = 2x W_k - W_{k-1}.
#[cube(launch_unchecked)]
pub fn kernel_chebyshev_w<F: Float>(x: &Array<F>, out: &mut Array<F>, n: u32) {
    if ABSOLUTE_POS < out.len() {
        let xv = x[ABSOLUTE_POS];
        let two_x = F::new(2.0) * xv;
        let n_u = n as usize;
        let mut prev2 = F::new(1.0);
        let mut prev1 = two_x + F::new(1.0);
        if n_u == 0 {
            out[ABSOLUTE_POS] = prev2;
        } else if n_u == 1 {
            out[ABSOLUTE_POS] = prev1;
        } else {
            for _ in 2..=n_u {
                let next = two_x * prev1 - prev2;
                prev2 = prev1;
                prev1 = next;
            }
            out[ABSOLUTE_POS] = prev1;
        }
    }
}

/// Hermite (physicist) H_n(x): H_0=1, H_1=2x, H_{k+1} = 2x H_k - 2k H_{k-1}.
#[cube(launch_unchecked)]
pub fn kernel_hermite_h<F: Float>(x: &Array<F>, out: &mut Array<F>, n: u32) {
    if ABSOLUTE_POS < out.len() {
        let xv = x[ABSOLUTE_POS];
        let two_x = F::new(2.0) * xv;
        let n_u = n as usize;
        let mut prev2 = F::new(1.0);
        let mut prev1 = two_x;
        if n_u == 0 {
            out[ABSOLUTE_POS] = prev2;
        } else if n_u == 1 {
            out[ABSOLUTE_POS] = prev1;
        } else {
            for k in 1..n_u {
                let kf = F::cast_from(k as u32);
                let next = two_x * prev1 - F::new(2.0) * kf * prev2;
                prev2 = prev1;
                prev1 = next;
            }
            out[ABSOLUTE_POS] = prev1;
        }
    }
}

/// Hermite (probabilist) He_n(x): He_0=1, He_1=x, He_{k+1} = x He_k - k He_{k-1}.
#[cube(launch_unchecked)]
pub fn kernel_hermite_he<F: Float>(x: &Array<F>, out: &mut Array<F>, n: u32) {
    if ABSOLUTE_POS < out.len() {
        let xv = x[ABSOLUTE_POS];
        let n_u = n as usize;
        let mut prev2 = F::new(1.0);
        let mut prev1 = xv;
        if n_u == 0 {
            out[ABSOLUTE_POS] = prev2;
        } else if n_u == 1 {
            out[ABSOLUTE_POS] = prev1;
        } else {
            for k in 1..n_u {
                let kf = F::cast_from(k as u32);
                let next = xv * prev1 - kf * prev2;
                prev2 = prev1;
                prev1 = next;
            }
            out[ABSOLUTE_POS] = prev1;
        }
    }
}

/// Laguerre L_n(x): L_0=1, L_1=1-x, (k+1) L_{k+1} = (2k+1-x) L_k - k L_{k-1}.
#[cube(launch_unchecked)]
pub fn kernel_laguerre_l<F: Float>(x: &Array<F>, out: &mut Array<F>, n: u32) {
    if ABSOLUTE_POS < out.len() {
        let xv = x[ABSOLUTE_POS];
        let n_u = n as usize;
        let mut prev2 = F::new(1.0);
        let mut prev1 = F::new(1.0) - xv;
        if n_u == 0 {
            out[ABSOLUTE_POS] = prev2;
        } else if n_u == 1 {
            out[ABSOLUTE_POS] = prev1;
        } else {
            for k in 1..n_u {
                let kf = F::cast_from(k as u32);
                let two_k_plus_one = F::new(2.0) * kf + F::new(1.0);
                let denom = kf + F::new(1.0);
                let next = ((two_k_plus_one - xv) * prev1 - kf * prev2) / denom;
                prev2 = prev1;
                prev1 = next;
            }
            out[ABSOLUTE_POS] = prev1;
        }
    }
}

/// Legendre P_n(x): P_0=1, P_1=x, (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}.
#[cube(launch_unchecked)]
pub fn kernel_legendre_p<F: Float>(x: &Array<F>, out: &mut Array<F>, n: u32) {
    if ABSOLUTE_POS < out.len() {
        let xv = x[ABSOLUTE_POS];
        let n_u = n as usize;
        let mut prev2 = F::new(1.0);
        let mut prev1 = xv;
        if n_u == 0 {
            out[ABSOLUTE_POS] = prev2;
        } else if n_u == 1 {
            out[ABSOLUTE_POS] = prev1;
        } else {
            for k in 1..n_u {
                let kf = F::cast_from(k as u32);
                let two_k_plus_one = F::new(2.0) * kf + F::new(1.0);
                let denom = kf + F::new(1.0);
                let next = (two_k_plus_one * xv * prev1 - kf * prev2) / denom;
                prev2 = prev1;
                prev1 = next;
            }
            out[ABSOLUTE_POS] = prev1;
        }
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
//
// Launch geometry (`elementwise_launch_dims`) lives in `crate::` (lib.rs);
// see that helper's doc comment. The function is shared verbatim with
// `quant.rs` and `grammar.rs`.

// ---------------------------------------------------------------------------
// Unary + binary helpers shared by every elementwise op
// ---------------------------------------------------------------------------

/// Upload `x`, launch `launcher`, return the on-device result handle and
/// element count.  No host readback is performed; the caller decides when
/// to call `client.read_one`. ADR #663 item 4.
fn run_unary<R, L>(
    client: &ComputeClient<R>,
    x: &[f32],
    launcher: L,
) -> (cubecl::server::Handle, usize)
where
    R: Runtime,
    L: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>),
{
    let n = x.len();
    let size_bytes = std::mem::size_of_val(x);

    let x_handle = client.create_from_slice(f32::as_bytes(x));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = crate::elementwise_launch_dims(n as u32);
    // SAFETY: `x_handle` was alloc'd by `client.create_from_slice` at line 399
    //   from `f32::as_bytes(x)`; backing buffer holds exactly `n = x.len()`
    //   f32 elements (line 396). `ArrayArg::from_raw_parts`'s second arg is
    //   the element count, not bytes; passing `n` matches the kernel's
    //   `&Array<f32>` view. The launcher (closure provided by caller) only
    //   dispatches kernels that read up to `n` elements — guarded by
    //   `ABSOLUTE_POS < out.len()` in every kernel body in this module.
    let in_arg = unsafe { ArrayArg::from_raw_parts(x_handle, n) };
    // SAFETY: `out_handle` was alloc'd by `client.empty(size_bytes)` at line
    //   400 with `size_bytes = n * size_of::<f32>()` (line 397 via
    //   `size_of_val`); capacity is exactly `n` f32 elements. `.clone()` on
    //   a cubecl `Handle` is a refcount bump, not a memory copy — both
    //   clones reference the same device allocation, so the kernel writes
    //   visible to `out_handle` (returned below). `n` is the element count.
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) };
    launcher(client, count, dim, in_arg, out_arg);

    (out_handle, n)
}

/// Upload `a` and `b`, launch `launcher`, return the on-device result handle
/// and element count.  No host readback is performed. ADR #663 item 4.
fn run_binary<R, L>(
    client: &ComputeClient<R>,
    a: &[f32],
    b: &[f32],
    launcher: L,
) -> (cubecl::server::Handle, usize)
where
    R: Runtime,
    L: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>, ArrayArg<R>),
{
    let n = a.len();
    debug_assert_eq!(n, b.len());
    let size_bytes = std::mem::size_of_val(a);

    let a_handle = client.create_from_slice(f32::as_bytes(a));
    let b_handle = client.create_from_slice(f32::as_bytes(b));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = crate::elementwise_launch_dims(n as u32);
    // SAFETY: `a_handle` was alloc'd by `client.create_from_slice` at line
    //   426 from `f32::as_bytes(a)`; backing buffer holds exactly
    //   `n = a.len()` f32 elements (line 422). `n == b.len()` is asserted
    //   at line 423. `ArrayArg::from_raw_parts`'s second arg is the element
    //   count; `n` matches the kernel's `&Array<f32>` first input.
    let a_arg = unsafe { ArrayArg::from_raw_parts(a_handle, n) };
    // SAFETY: `b_handle` was alloc'd by `client.create_from_slice` at line
    //   427 from `f32::as_bytes(b)`; backing buffer holds exactly
    //   `b.len() == n` f32 elements (debug-asserted line 423). `n` matches
    //   the kernel's `&Array<f32>` second input.
    let b_arg = unsafe { ArrayArg::from_raw_parts(b_handle, n) };
    // SAFETY: `out_handle` was alloc'd by `client.empty(size_bytes)` at line
    //   428 with `size_bytes = n * size_of::<f32>()` (line 424 via
    //   `size_of_val`); capacity is exactly `n` f32 elements. `.clone()` is
    //   a cubecl handle refcount bump (no copy), so the kernel writes are
    //   visible through the returned `out_handle`. `n` is the element count.
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) };
    launcher(client, count, dim, a_arg, b_arg, out_arg);

    (out_handle, n)
}

// ---------------------------------------------------------------------------
// Handle-direct helpers — no host upload for already device-resident data
// ---------------------------------------------------------------------------

/// Run a unary kernel against a pre-uploaded device handle.
///
/// Accepts a `cubecl::server::Handle` that already points to GPU memory;
/// no `create_from_slice` is called. Used by the XPU path after #673 so
/// inputs that are already device-resident don't round-trip through host.
fn run_unary_handle<R, L>(
    client: &ComputeClient<R>,
    x_handle: cubecl::server::Handle,
    n: usize,
    launcher: L,
) -> (cubecl::server::Handle, usize)
where
    R: Runtime,
    L: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>),
{
    let size_bytes = n * std::mem::size_of::<f32>();
    let out_handle = client.empty(size_bytes);
    let (count, dim) = crate::elementwise_launch_dims(n as u32);
    crate::debug_assert_handle_capacity::<f32>(&x_handle, n);
    // SAFETY: `x_handle` and `out_handle` were allocated by this `client`
    // with exactly `n` f32 elements each. The kernel reads `n` elements from
    // `x_handle` and writes `n` elements to `out_handle`. Verified at
    // debug-build runtime via `debug_assert_handle_capacity::<f32>(&x_handle,
    // n)` immediately above; release builds rely on the caller contract.
    let in_arg = unsafe { ArrayArg::from_raw_parts(x_handle, n) };
    // SAFETY: `out_handle` is the freshly-alloc'd device buffer from line
    //   486 (`client.empty(n * size_of::<f32>())`); capacity is exactly `n`
    //   f32 elements. `.clone()` on a cubecl `Handle` is a refcount bump
    //   only, so the kernel writes through this `ArrayArg` are visible via
    //   the `out_handle` returned to the caller. `n` is the element count
    //   (not bytes), matching the kernel's `&mut Array<f32>` view.
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) };
    launcher(client, count, dim, in_arg, out_arg);
    (out_handle, n)
}

/// Run a binary kernel against two pre-uploaded device handles.
///
/// Same as `run_unary_handle` but for two inputs. No host upload. Issue #673.
fn run_binary_handle<R, L>(
    client: &ComputeClient<R>,
    a_handle: cubecl::server::Handle,
    b_handle: cubecl::server::Handle,
    n: usize,
    launcher: L,
) -> (cubecl::server::Handle, usize)
where
    R: Runtime,
    L: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>, ArrayArg<R>),
{
    let size_bytes = n * std::mem::size_of::<f32>();
    let out_handle = client.empty(size_bytes);
    let (count, dim) = crate::elementwise_launch_dims(n as u32);
    crate::debug_assert_handle_capacity::<f32>(&a_handle, n);
    // SAFETY: `a_handle`, `b_handle`, and `out_handle` were allocated by this
    // `client` with exactly `n` f32 elements each. The kernel reads from the
    // first two and writes to the third. Verified at debug-build runtime via
    // `debug_assert_handle_capacity::<f32>(&a_handle, n)` immediately above;
    // release builds rely on the caller contract.
    let a_arg = unsafe { ArrayArg::from_raw_parts(a_handle, n) };
    crate::debug_assert_handle_capacity::<f32>(&b_handle, n);
    // SAFETY: `b_handle` is caller-provided and contracted to be alloc'd
    //   from this same `client` with ≥`n * size_of::<f32>()` bytes (handle
    //   shape is the `pub fn` contract for the `*_handle` runner family —
    //   see issue #673). `n` matches the kernel's `&Array<f32>` second
    //   input element count. Verified at debug-build runtime via
    //   `debug_assert_handle_capacity::<f32>(&b_handle, n)` immediately above;
    //   release builds rely on the caller contract.
    let b_arg = unsafe { ArrayArg::from_raw_parts(b_handle, n) };
    // SAFETY: `out_handle` is the freshly-alloc'd device buffer from line
    //   512 (`client.empty(n * size_of::<f32>())`); capacity is exactly `n`
    //   f32 elements. `.clone()` is a refcount bump only — kernel writes
    //   through this `ArrayArg` are visible via the returned `out_handle`.
    //   `n` is the element count, matching the kernel's
    //   `&mut Array<f32>` view.
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) };
    launcher(client, count, dim, a_arg, b_arg, out_arg);
    (out_handle, n)
}

// Per-kernel `run_*` helpers: each one is just a thin wrapper around
// `run_unary` / `run_binary` that plugs in the generated
// `kernel_*::launch_unchecked` symbol. Two macros stamp them out.
// Each also gains a `*_handle` variant (no H2D upload) for the XPU path.

macro_rules! define_unary_runner {
    ($run_fn:ident, $run_fn_handle:ident, $kernel:ident) => {
        #[doc = concat!("Upload `x`, run `", stringify!($kernel), "`, return on-device handle + element count. ADR #663 item 4.")]
        pub fn $run_fn<R: Runtime>(
            client: &ComputeClient<R>,
            x: &[f32],
        ) -> (cubecl::server::Handle, usize) {
            // SAFETY: `launch_unchecked` is unsafe per cubecl convention because
            //   it skips runtime arity/dim sanity checks that the macro form
            //   `launch` performs. Caller (`run_unary`) guarantees: `input` and
            //   `output` are `ArrayArg::from_raw_parts` over `n = x.len()` f32
            //   elements (constructed under SAFETY at lines 411-426); `count`
            //   and `dim` come from `elementwise_launch_dims(n as u32)` so the
            //   grid covers exactly `n` units (one per element); the kernel
            //   body guards `ABSOLUTE_POS < out.len()`. Argument refs live for
            //   the launch duration; cubecl queues the dispatch and returns,
            //   sync is the caller's responsibility.
            run_unary::<R, _>(client, x, |client, count, dim, input, output| unsafe {
                $kernel::launch_unchecked::<f32, R>(client, count, dim, input, output);
            })
        }

        #[doc = concat!("Run `", stringify!($kernel), "` on a pre-uploaded device handle; no H2D upload. Issue #673.")]
        pub fn $run_fn_handle<R: Runtime>(
            client: &ComputeClient<R>,
            x_handle: cubecl::server::Handle,
            n: usize,
        ) -> (cubecl::server::Handle, usize) {
            // SAFETY: same `launch_unchecked` invariants as the slice-upload
            //   variant above. `run_unary_handle` (line 448) wraps `x_handle`
            //   and a freshly-allocated `out_handle` in `ArrayArg::from_raw_parts`
            //   over `n` f32 elements. The handle-direct path requires the
            //   caller to have alloc'd `x_handle` from the same `client` with
            //   ≥`n * size_of::<f32>()` bytes — this is the `pub fn` contract
            //   used by `ferrotorch-xpu` after #673. `count`/`dim` cover `n`
            //   units; kernel body bounds-checks `ABSOLUTE_POS`.
            run_unary_handle::<R, _>(client, x_handle, n, |client, count, dim, input, output| unsafe {
                $kernel::launch_unchecked::<f32, R>(client, count, dim, input, output);
            })
        }
    };
}

macro_rules! define_binary_runner {
    ($run_fn:ident, $run_fn_handle:ident, $kernel:ident) => {
        #[doc = concat!("Upload `a` and `b`, run `", stringify!($kernel), "`, return on-device handle + element count. ADR #663 item 4.")]
        pub fn $run_fn<R: Runtime>(
            client: &ComputeClient<R>,
            a: &[f32],
            b: &[f32],
        ) -> (cubecl::server::Handle, usize) {
            // SAFETY: `launch_unchecked` skips arity/dim checks; safety is
            //   inherited from `run_binary` (line 412), which constructs all
            //   three `ArrayArg`s under SAFETY at lines 439-457: `a` and `b`
            //   handles each hold `n = a.len() == b.len()` f32 elements
            //   (asserted line 423), `out` handle holds `n` elements freshly
            //   alloc'd. `count`/`dim` from `elementwise_launch_dims(n as u32)`
            //   span exactly `n` units. Kernel body bounds-checks
            //   `ABSOLUTE_POS < out.len()`. Refs live for launch duration;
            //   cubecl queues + returns.
            run_binary::<R, _>(client, a, b, |client, count, dim, a, b, out| unsafe {
                $kernel::launch_unchecked::<f32, R>(client, count, dim, a, b, out);
            })
        }

        #[doc = concat!("Run `", stringify!($kernel), "` on pre-uploaded device handles; no H2D upload. Issue #673.")]
        pub fn $run_fn_handle<R: Runtime>(
            client: &ComputeClient<R>,
            a_handle: cubecl::server::Handle,
            b_handle: cubecl::server::Handle,
            n: usize,
        ) -> (cubecl::server::Handle, usize) {
            // SAFETY: same `launch_unchecked` invariants as the slice-upload
            //   binary variant above. `run_binary_handle` (line 473) wraps
            //   the two caller-provided handles plus a fresh `out_handle` in
            //   `ArrayArg::from_raw_parts` over `n` f32 elements. The
            //   `pub fn` contract requires both `a_handle` and `b_handle` to
            //   be alloc'd from the same `client` with ≥`n * size_of::<f32>()`
            //   bytes (caller responsibility — used by `ferrotorch-xpu`
            //   post-#673). Grid covers `n` units; kernel guards
            //   `ABSOLUTE_POS < out.len()`.
            run_binary_handle::<R, _>(client, a_handle, b_handle, n, |client, count, dim, a, b, out| unsafe {
                $kernel::launch_unchecked::<f32, R>(client, count, dim, a, b, out);
            })
        }
    };
}

// Binary ops — slice-upload variant + handle-direct variant (issue #673)
define_binary_runner!(run_add, run_add_handle, kernel_add);
define_binary_runner!(run_sub, run_sub_handle, kernel_sub);
define_binary_runner!(run_mul, run_mul_handle, kernel_mul);
define_binary_runner!(run_div, run_div_handle, kernel_div);

// Unary ops — slice-upload variant + handle-direct variant (issue #673)
define_unary_runner!(run_relu, run_relu_handle, kernel_relu);
define_unary_runner!(run_neg, run_neg_handle, kernel_neg);
define_unary_runner!(run_abs, run_abs_handle, kernel_abs);
define_unary_runner!(run_exp, run_exp_handle, kernel_exp);
define_unary_runner!(run_ln, run_ln_handle, kernel_ln);
define_unary_runner!(run_sqrt, run_sqrt_handle, kernel_sqrt);
define_unary_runner!(run_sin, run_sin_handle, kernel_sin);
define_unary_runner!(run_cos, run_cos_handle, kernel_cos);
define_unary_runner!(run_tanh, run_tanh_handle, kernel_tanh);
define_unary_runner!(run_sigmoid, run_sigmoid_handle, kernel_sigmoid);

// ---------------------------------------------------------------------------
// Polynomial runners — unary + scalar `n` (degree). (#577)
// ---------------------------------------------------------------------------

/// Run a unary polynomial kernel taking an extra `n: u32` (degree) scalar.
/// Same pattern as `run_unary` but threads through a single scalar argument.
/// Returns the on-device handle and element count; no host readback. ADR #663.
fn run_unary_with_n<R, L>(
    client: &ComputeClient<R>,
    x: &[f32],
    n: u32,
    launcher: L,
) -> (cubecl::server::Handle, usize)
where
    R: Runtime,
    L: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>, u32),
{
    let count_elems = x.len();
    let size_bytes = std::mem::size_of_val(x);

    let x_handle = client.create_from_slice(f32::as_bytes(x));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = crate::elementwise_launch_dims(count_elems as u32);
    // SAFETY: `x_handle` was alloc'd by `client.create_from_slice` at line
    //   655 from `f32::as_bytes(x)`; backing buffer holds exactly
    //   `count_elems = x.len()` f32 elements (line 652).
    //   `ArrayArg::from_raw_parts`'s second arg is the element count;
    //   `count_elems` matches the kernel's `&Array<f32>` input view. The
    //   scalar `n` (degree) is passed separately and not subject to this
    //   block's invariants.
    let in_arg = unsafe { ArrayArg::from_raw_parts(x_handle, count_elems) };
    // SAFETY: `out_handle` was alloc'd by `client.empty(size_bytes)` at line
    //   656 with `size_bytes = count_elems * size_of::<f32>()` (line 653 via
    //   `size_of_val`); capacity is exactly `count_elems` f32 elements.
    //   `.clone()` is a cubecl handle refcount bump (no copy); kernel
    //   writes are visible through the returned `out_handle`. `count_elems`
    //   is the element count.
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), count_elems) };
    launcher(client, count, dim, in_arg, out_arg, n);

    (out_handle, count_elems)
}

/// Run a unary polynomial kernel against a pre-uploaded device handle,
/// taking an extra `degree: u32` scalar.
///
/// Same pattern as [`run_unary_handle`] but threads through one extra scalar
/// argument (the polynomial degree). No host upload — the input handle is
/// caller-provided and contracted to be alloc'd from this same `client` with
/// `≥ n * size_of::<f32>()` bytes. Used by the XPU path so already
/// device-resident polynomial inputs don't round-trip through host. Issue
/// #715 (#673 cascade).
fn run_unary_with_n_handle<R, L>(
    client: &ComputeClient<R>,
    x_handle: cubecl::server::Handle,
    n: usize,
    degree: u32,
    launcher: L,
) -> (cubecl::server::Handle, usize)
where
    R: Runtime,
    L: FnOnce(&ComputeClient<R>, CubeCount, CubeDim, ArrayArg<R>, ArrayArg<R>, u32),
{
    let size_bytes = n * std::mem::size_of::<f32>();
    let out_handle = client.empty(size_bytes);
    let (count, dim) = crate::elementwise_launch_dims(n as u32);
    crate::debug_assert_handle_capacity::<f32>(&x_handle, n);
    // SAFETY: `x_handle` is caller-provided and contracted (the `pub fn`
    //   contract for the `*_handle` runner family — see issue #715) to be
    //   alloc'd from this same `client` with `≥ n * size_of::<f32>()` bytes.
    //   `ArrayArg::from_raw_parts`'s second arg is the element count (not
    //   bytes); `n` matches the kernel's `&Array<f32>` input view. The
    //   capacity precondition is verified at debug-build runtime via
    //   `debug_assert_handle_capacity::<f32>(&x_handle, n)` immediately above
    //   (release builds rely on the caller contract). The scalar `degree` is
    //   passed by value through the launcher and is not subject to this
    //   block's handle-aliasing invariants. The kernel guards
    //   `ABSOLUTE_POS < out.len()` so reads are bounds-checked at the
    //   per-unit level.
    let in_arg = unsafe { ArrayArg::from_raw_parts(x_handle, n) };
    // SAFETY: `out_handle` is the freshly-alloc'd device buffer from
    //   `client.empty(size_bytes)` at the top of this function with
    //   `size_bytes = n * size_of::<f32>()`; capacity is exactly `n` f32
    //   elements by construction. `.clone()` on a cubecl `Handle` is a
    //   refcount bump, not a memory copy — both clones reference the same
    //   device allocation, so kernel writes through this `ArrayArg` are
    //   visible via the `out_handle` returned to the caller. `n` is the
    //   element count, matching the kernel's `&mut Array<f32>` view; the
    //   output buffer is exclusively owned (no aliasing with `x_handle`).
    let out_arg = unsafe { ArrayArg::from_raw_parts(out_handle.clone(), n) };
    launcher(client, count, dim, in_arg, out_arg, degree);
    (out_handle, n)
}

macro_rules! define_unary_with_n_runner {
    ($run_fn:ident, $kernel:ident) => {
        #[doc = concat!("Upload `x`, run `", stringify!($kernel), "` with degree `n`, return on-device handle + element count. ADR #663 item 4.")]
        pub fn $run_fn<R: Runtime>(
            client: &ComputeClient<R>,
            x: &[f32],
            n: u32,
        ) -> (cubecl::server::Handle, usize) {
            run_unary_with_n::<R, _>(
                client,
                x,
                n,
                // SAFETY: `launch_unchecked` skips runtime arity/dim checks
                //   that the macro `launch` form normally performs. Caller
                //   (`run_unary_with_n`, line 642) constructs `input` and
                //   `output` via `ArrayArg::from_raw_parts` over
                //   `count_elems = x.len()` f32 elements (under SAFETY at
                //   lines 658-674). `count`/`dim` come from
                //   `elementwise_launch_dims(count_elems as u32)` so the grid
                //   covers `count_elems` units (one per element); kernel
                //   bodies bounds-check `ABSOLUTE_POS < out.len()`. `n_val`
                //   is a scalar copied by value into the kernel — not subject
                //   to handle-aliasing concerns. Refs live for launch
                //   duration; cubecl queues the dispatch and returns.
                |client, count, dim, input, output, n_val| unsafe {
                    $kernel::launch_unchecked::<f32, R>(
                        client,
                        count,
                        dim,
                        input,
                        output,
                        n_val,
                    );
                },
            )
        }
    };
}

macro_rules! define_unary_with_n_runner_handle {
    ($run_fn:ident, $kernel:ident) => {
        #[doc = concat!("Run `", stringify!($kernel), "` on a pre-uploaded device handle with degree `n`; no H2D upload. Issue #715.")]
        pub fn $run_fn<R: Runtime>(
            client: &ComputeClient<R>,
            x_handle: cubecl::server::Handle,
            n: usize,
            degree: u32,
        ) -> (cubecl::server::Handle, usize) {
            run_unary_with_n_handle::<R, _>(
                client,
                x_handle,
                n,
                degree,
                // SAFETY: same `launch_unchecked` invariants as the
                //   slice-upload `define_unary_with_n_runner!` stamp above.
                //   `run_unary_with_n_handle` wraps the caller-provided
                //   `x_handle` and a freshly-allocated `out_handle` in
                //   `ArrayArg::from_raw_parts` over `n` f32 elements (under
                //   SAFETY in that function's body). The handle-direct path
                //   requires the caller to have alloc'd `x_handle` from the
                //   same `client` with `≥ n * size_of::<f32>()` bytes — the
                //   `pub fn` contract used by `ferrotorch-xpu` post-#715,
                //   verified at debug-build runtime by
                //   `debug_assert_handle_capacity::<f32>` inside
                //   `run_unary_with_n_handle`. `count`/`dim` cover `n`
                //   units; kernel body bounds-checks `ABSOLUTE_POS`. `n_val`
                //   is a scalar copied by value into the kernel and not
                //   subject to handle-aliasing concerns.
                |client, count, dim, input, output, n_val| unsafe {
                    $kernel::launch_unchecked::<f32, R>(
                        client,
                        count,
                        dim,
                        input,
                        output,
                        n_val,
                    );
                },
            )
        }
    };
}

define_unary_with_n_runner!(run_chebyshev_t, kernel_chebyshev_t);
define_unary_with_n_runner!(run_chebyshev_u, kernel_chebyshev_u);
define_unary_with_n_runner!(run_chebyshev_v, kernel_chebyshev_v);
define_unary_with_n_runner!(run_chebyshev_w, kernel_chebyshev_w);
define_unary_with_n_runner!(run_hermite_h, kernel_hermite_h);
define_unary_with_n_runner!(run_hermite_he, kernel_hermite_he);
define_unary_with_n_runner!(run_laguerre_l, kernel_laguerre_l);
define_unary_with_n_runner!(run_legendre_p, kernel_legendre_p);

// Handle-direct variants (no H2D upload) — used by `ferrotorch-xpu` after
// #715 when polynomial inputs are already device-resident.
define_unary_with_n_runner_handle!(run_chebyshev_t_handle, kernel_chebyshev_t);
define_unary_with_n_runner_handle!(run_chebyshev_u_handle, kernel_chebyshev_u);
define_unary_with_n_runner_handle!(run_chebyshev_v_handle, kernel_chebyshev_v);
define_unary_with_n_runner_handle!(run_chebyshev_w_handle, kernel_chebyshev_w);
define_unary_with_n_runner_handle!(run_hermite_h_handle, kernel_hermite_h);
define_unary_with_n_runner_handle!(run_hermite_he_handle, kernel_hermite_he);
define_unary_with_n_runner_handle!(run_laguerre_l_handle, kernel_laguerre_l);
define_unary_with_n_runner_handle!(run_legendre_p_handle, kernel_legendre_p);

/// Upload `a` and `b`, run `kernel_matmul_naive`, return the on-device result
/// handle and element count (`m * n`).  No host readback is performed.
///
/// `a` is `[m * k]` row-major, `b` is `[k * n]` row-major, output is
/// `[m * n]` row-major. The caller is responsible for verifying sizes.
/// ADR #663 item 4.
pub fn run_matmul<R: Runtime>(
    client: &ComputeClient<R>,
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> (cubecl::server::Handle, usize) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    let out_len = m * n;
    let size_bytes = out_len * std::mem::size_of::<f32>();

    let a_handle = client.create_from_slice(f32::as_bytes(a));
    let b_handle = client.create_from_slice(f32::as_bytes(b));
    let out_handle = client.empty(size_bytes);

    let (count, dim) = crate::elementwise_launch_dims(out_len as u32);
    // SAFETY: All three handles were alloc'd by this `client`:
    //   - `a_handle` from `create_from_slice(f32::as_bytes(a))` at line 746;
    //     backing buffer holds `a.len() == m * k` f32 elements (asserted
    //     line 741). Passing `a.len()` matches the kernel's `&Array<f32>`
    //     first input.
    //   - `b_handle` from `create_from_slice(f32::as_bytes(b))` at line 747;
    //     `b.len() == k * n` f32 elements (asserted line 742). `b.len()`
    //     matches the second input.
    //   - `out_handle` from `empty(size_bytes)` at line 748 with
    //     `size_bytes = out_len * size_of::<f32>()` (line 744); capacity is
    //     exactly `out_len = m * n` f32 elements. `.clone()` is a refcount
    //     bump, kernel writes are visible through the returned handle.
    //   `count`/`dim` from `elementwise_launch_dims(out_len)` cover `m*n`
    //   units (one per output element); kernel guards `ABSOLUTE_POS <
    //   out.len()`. Scalars `m`, `k`, `n` are passed by value as `u32` for
    //   shape indexing inside the kernel — kernel converts to `usize`
    //   internally per the documented idiom (see `kernel_matmul_naive`).
    //   `launch_unchecked` is unsafe per cubecl convention because it
    //   bypasses runtime arity checks; refs live for launch duration;
    //   cubecl queues the dispatch.
    unsafe {
        kernel_matmul_naive::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(a_handle, a.len()),
            ArrayArg::from_raw_parts(b_handle, b.len()),
            ArrayArg::from_raw_parts(out_handle.clone(), out_len),
            m as u32,
            k as u32,
            n as u32,
        );
    }

    (out_handle, out_len)
}

/// Run `kernel_matmul_naive` against pre-uploaded device handles; no H2D
/// upload. Issue #673.
///
/// `a_handle` points to `[m * k]` row-major device memory and `b_handle`
/// points to `[k * n]`. Output is allocated by this function and filled in
/// `[m * n]` row-major. The caller is responsible for ensuring the handle
/// shapes and sizes match `m`, `k`, `n`.
///
/// This is the device-resident counterpart to [`run_matmul`] for callers
/// whose inputs already live on the GPU (e.g. `ferrotorch-xpu` after #673).
pub fn run_matmul_handle<R: Runtime>(
    client: &ComputeClient<R>,
    a_handle: cubecl::server::Handle,
    b_handle: cubecl::server::Handle,
    m: usize,
    k: usize,
    n: usize,
) -> (cubecl::server::Handle, usize) {
    let a_len = m * k;
    let b_len = k * n;
    let out_len = m * n;
    let size_bytes = out_len * std::mem::size_of::<f32>();

    let out_handle = client.empty(size_bytes);
    let (count, dim) = crate::elementwise_launch_dims(out_len as u32);
    crate::debug_assert_handle_capacity::<f32>(&a_handle, a_len);
    crate::debug_assert_handle_capacity::<f32>(&b_handle, b_len);
    // SAFETY: `a_handle`, `b_handle`, and `out_handle` were allocated by
    // this `client`. The kernel reads `a_len = m*k` and `b_len = k*n`
    // elements and writes `out_len = m*n` elements — matching the array
    // lengths declared here. Verified at debug-build runtime via
    // `debug_assert_handle_capacity::<f32>(&a_handle, a_len)` and
    // `debug_assert_handle_capacity::<f32>(&b_handle, b_len)` immediately
    // above; release builds rely on the caller contract.
    unsafe {
        kernel_matmul_naive::launch_unchecked::<f32, R>(
            client,
            count,
            dim,
            ArrayArg::from_raw_parts(a_handle, a_len),
            ArrayArg::from_raw_parts(b_handle, b_len),
            ArrayArg::from_raw_parts(out_handle.clone(), out_len),
            m as u32,
            k as u32,
            n as u32,
        );
    }

    (out_handle, out_len)
}
