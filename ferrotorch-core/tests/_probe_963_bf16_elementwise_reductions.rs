//! Permanent regression sentinel for #963: bf16 elementwise, reduction, and
//! activation GPU ops (add, sub, mul, div, sum_axis, mean_axis, relu, sigmoid).
//!
//! Pre-fix observable failure (before #963):
//!   * `GpuBackend::add_bf16_f32`, `sub_bf16_f32`, `mul_bf16_f32`,
//!     `div_bf16_f32`, `sum_axis_bf16_f32`, `mean_axis_bf16_f32`,
//!     `relu_bf16_f32`, `sigmoid_bf16_f32` all returned
//!     `Err(InvalidArgument { "... not implemented ..." })` — the default
//!     trait-method stub — so any autocast bf16 elementwise/reduction path
//!     on CUDA was a hard error rather than a GPU kernel.
//!
//! Post-fix:
//!   * Each op launches a PTX kernel (sm_52+) on the RTX 3090.
//!   * bf16 inputs are decoded on-device: ld.global.u16 -> zero-extend ->
//!     shift-left-16 -> reinterpret as f32 (the standard bf16 bit layout).
//!   * All arithmetic and accumulation is in f32.
//!   * f32 output matches the CPU reference within F32_BF16_TOL = 1e-3
//!     (bf16 has ~7 mantissa bits; 1e-3 is the documented C.3 tolerance).
//!
//! PyTorch parity (rust-gpu-discipline §3):
//!   `torch.add(a.bfloat16(), b.bfloat16())` etc. under `torch.autocast`
//!   on CUDA produce f32 outputs with internal f32 arithmetic.
//!   relu and sigmoid on bf16 tensors follow the same autocast contract.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::gpu_dispatch;

// bf16 has ~7 mantissa bits => ~0.78% relative error per value.
// For elementwise ops with no accumulation, 1e-3 absolute is sufficient.
// For reductions over a small axis (axis_size <= 8 in tests), accumulated
// error is still within 1e-3.
const F32_BF16_TOL: f32 = 1e-3;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for bf16 elementwise probe");
    });
}

// ---------------------------------------------------------------------------
// Upload / download helpers
// ---------------------------------------------------------------------------

/// Convert f32 to bf16 bit pattern (round-to-nearest-even).
fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let round = bits.wrapping_add(0x7FFF).wrapping_add(lsb);
    (round >> 16) as u16
}

/// Upload f32 slice as bf16 bit patterns (CudaSlice<u16> handle, elem_size=2).
fn upload_bf16(data: &[f32]) -> gpu_dispatch::GpuBufferHandle {
    let bf16_vals: Vec<u16> = data.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(bf16_vals.as_ptr() as *const u8, bf16_vals.len() * 2)
    };
    let backend = gpu_dispatch::gpu_backend().expect("backend");
    backend.cpu_to_gpu(bytes, 2, 0).expect("upload bf16")
}

/// Download f32 GPU handle.
fn download_f32(h: &gpu_dispatch::GpuBufferHandle) -> Vec<f32> {
    let backend = gpu_dispatch::gpu_backend().expect("backend");
    let bytes = backend.gpu_to_cpu(h).expect("download f32");
    let vals: &[f32] =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) };
    vals.to_vec()
}

fn check(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_BF16_TOL,
            "{label} elem {i}: got {g}, expected {e} (tol {F32_BF16_TOL})"
        );
    }
}

// ---------------------------------------------------------------------------
// #963: add_bf16_f32
// ---------------------------------------------------------------------------

/// BEFORE: add_bf16_f32 returned Err("not implemented").
/// AFTER:  PTX add_bf16_f32_kernel on GPU; f32 output within 1e-3.
#[test]
fn p963_add_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.5f32, 1.5, 2.5, 3.5];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();

    let a_gpu = upload_bf16(&a);
    let b_gpu = upload_bf16(&b);

    let result = backend
        .add_bf16_f32(&a_gpu, &b_gpu, 4)
        .expect("add_bf16_f32 must succeed on CUDA");

    assert_eq!(result.len(), 4);
    assert_eq!(result.device_ordinal(), 0);
    check("add_bf16_f32_basic", &download_f32(&result), &expected);
}

#[test]
fn p963_add_bf16_f32_zero_len() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");
    let a = upload_bf16(&[]);
    let b = upload_bf16(&[]);
    let r = backend.add_bf16_f32(&a, &b, 0).expect("zero-len add must not error");
    assert_eq!(r.len(), 0);
}

// ---------------------------------------------------------------------------
// #963: sub_bf16_f32
// ---------------------------------------------------------------------------

/// BEFORE: sub_bf16_f32 returned Err("not implemented").
/// AFTER:  PTX sub_bf16_f32_kernel on GPU; f32 output within 1e-3.
#[test]
fn p963_sub_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a = vec![5.0f32, 4.0, 3.0, 2.0];
    let b = vec![1.0f32, 1.5, 0.5, 0.25];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();

    let a_gpu = upload_bf16(&a);
    let b_gpu = upload_bf16(&b);
    let result = backend
        .sub_bf16_f32(&a_gpu, &b_gpu, 4)
        .expect("sub_bf16_f32 must succeed on CUDA");
    check("sub_bf16_f32_basic", &download_f32(&result), &expected);
}

// ---------------------------------------------------------------------------
// #963: mul_bf16_f32
// ---------------------------------------------------------------------------

/// BEFORE: mul_bf16_f32 returned Err("not implemented").
/// AFTER:  PTX mul_bf16_f32_kernel on GPU; f32 output within 1e-3.
#[test]
fn p963_mul_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a = vec![2.0f32, 3.0, 0.5, 1.0];
    let b = vec![4.0f32, 2.0, 2.0, 0.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();

    let a_gpu = upload_bf16(&a);
    let b_gpu = upload_bf16(&b);
    let result = backend
        .mul_bf16_f32(&a_gpu, &b_gpu, 4)
        .expect("mul_bf16_f32 must succeed on CUDA");
    check("mul_bf16_f32_basic", &download_f32(&result), &expected);
}

// ---------------------------------------------------------------------------
// #963: div_bf16_f32
// ---------------------------------------------------------------------------

/// BEFORE: div_bf16_f32 returned Err("not implemented").
/// AFTER:  PTX div_bf16_f32_kernel on GPU; f32 output within 1e-3.
#[test]
fn p963_div_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a = vec![6.0f32, 4.0, 9.0, 1.0];
    let b = vec![2.0f32, 4.0, 3.0, 1.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect();

    let a_gpu = upload_bf16(&a);
    let b_gpu = upload_bf16(&b);
    let result = backend
        .div_bf16_f32(&a_gpu, &b_gpu, 4)
        .expect("div_bf16_f32 must succeed on CUDA");
    check("div_bf16_f32_basic", &download_f32(&result), &expected);
}

// ---------------------------------------------------------------------------
// #963: sum_axis_bf16_f32
// ---------------------------------------------------------------------------

/// BEFORE: sum_axis_bf16_f32 returned Err("not implemented").
/// AFTER:  PTX sum_axis_bf16_f32_kernel on GPU; f32 output within 1e-3.
///
/// Layout: [outer=2, axis=4, inner=1] -> output [2, 1].
/// Row 0: 1+2+3+4 = 10. Row 1: 5+6+7+8 = 26.
#[test]
fn p963_sum_axis_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    // [2, 4, 1] = outer=2, axis=4, inner=1
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected = vec![10.0f32, 26.0];

    let a_gpu = upload_bf16(&data);
    let result = backend
        .sum_axis_bf16_f32(&a_gpu, 2, 4, 1)
        .expect("sum_axis_bf16_f32 must succeed on CUDA");

    assert_eq!(result.len(), 2);
    assert_eq!(result.device_ordinal(), 0);
    check("sum_axis_bf16_f32_basic", &download_f32(&result), &expected);
}

/// Inner dim = 2: layout [outer=1, axis=3, inner=2] -> output [1, 2].
/// Col 0: 1+3+5=9, Col 1: 2+4+6=12.
#[test]
fn p963_sum_axis_bf16_f32_inner_dim() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected = vec![9.0f32, 12.0];

    let a_gpu = upload_bf16(&data);
    let result = backend
        .sum_axis_bf16_f32(&a_gpu, 1, 3, 2)
        .expect("sum_axis_bf16_f32 inner_dim");
    check("sum_axis_bf16_f32_inner_dim", &download_f32(&result), &expected);
}

// ---------------------------------------------------------------------------
// #963: mean_axis_bf16_f32
// ---------------------------------------------------------------------------

/// BEFORE: mean_axis_bf16_f32 returned Err("not implemented").
/// AFTER:  PTX mean_axis_bf16_f32_kernel on GPU; f32 output within 1e-3.
///
/// Layout: [outer=2, axis=4, inner=1].
/// Row 0: (1+2+3+4)/4 = 2.5. Row 1: (5+6+7+8)/4 = 6.5.
#[test]
fn p963_mean_axis_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected = vec![2.5f32, 6.5];

    let a_gpu = upload_bf16(&data);
    let result = backend
        .mean_axis_bf16_f32(&a_gpu, 2, 4, 1)
        .expect("mean_axis_bf16_f32 must succeed on CUDA");

    assert_eq!(result.len(), 2);
    assert_eq!(result.device_ordinal(), 0);
    check("mean_axis_bf16_f32_basic", &download_f32(&result), &expected);
}

// ---------------------------------------------------------------------------
// #963: relu_bf16_f32
// ---------------------------------------------------------------------------

/// BEFORE: relu_bf16_f32 returned Err("not implemented").
/// AFTER:  PTX relu_bf16_f32_kernel on GPU; f32 output within 1e-3.
///
/// relu([-1, 0, 1, 2]) = [0, 0, 1, 2].
#[test]
fn p963_relu_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let x = vec![-1.0f32, 0.0, 1.0, 2.0];
    let expected = vec![0.0f32, 0.0, 1.0, 2.0];

    let x_gpu = upload_bf16(&x);
    let result = backend
        .relu_bf16_f32(&x_gpu, 4)
        .expect("relu_bf16_f32 must succeed on CUDA");

    assert_eq!(result.len(), 4);
    assert_eq!(result.device_ordinal(), 0);
    check("relu_bf16_f32_basic", &download_f32(&result), &expected);
}

/// All negative -> all zeros.
#[test]
fn p963_relu_bf16_f32_all_negative() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let x = vec![-5.0f32, -3.0, -1.0, -0.5];
    let expected = vec![0.0f32; 4];

    let x_gpu = upload_bf16(&x);
    let result = backend
        .relu_bf16_f32(&x_gpu, 4)
        .expect("relu all-negative");
    check("relu_bf16_f32_all_negative", &download_f32(&result), &expected);
}

// ---------------------------------------------------------------------------
// #963: sigmoid_bf16_f32
// ---------------------------------------------------------------------------

/// BEFORE: sigmoid_bf16_f32 returned Err("not implemented").
/// AFTER:  PTX sigmoid_bf16_f32_kernel on GPU; f32 output within 1e-3.
///
/// sigmoid(0) = 0.5 exactly.
/// sigmoid(large positive) -> ~1.0; sigmoid(large negative) -> ~0.0.
#[test]
fn p963_sigmoid_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let x = vec![0.0f32, 2.0, -2.0, 10.0, -10.0];
    let expected: Vec<f32> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();

    let x_gpu = upload_bf16(&x);
    let result = backend
        .sigmoid_bf16_f32(&x_gpu, 5)
        .expect("sigmoid_bf16_f32 must succeed on CUDA");

    assert_eq!(result.len(), 5);
    assert_eq!(result.device_ordinal(), 0);
    check("sigmoid_bf16_f32_basic", &download_f32(&result), &expected);
}

/// sigmoid(0) must equal 0.5 to within 1e-3.
#[test]
fn p963_sigmoid_bf16_f32_zero_input() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let x_gpu = upload_bf16(&[0.0f32]);
    let result = backend
        .sigmoid_bf16_f32(&x_gpu, 1)
        .expect("sigmoid zero input");
    let got = download_f32(&result);
    assert!(
        (got[0] - 0.5).abs() < 1e-3,
        "sigmoid(0) must be 0.5, got {}",
        got[0]
    );
}

/// Larger tensor: 128 elements, values in [-3, 3].
/// Exercises multi-block launch path.
#[test]
fn p963_sigmoid_bf16_f32_128_elements() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let n = 128usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 6.0 - 3.0).collect();
    let expected: Vec<f32> = x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect();

    let x_gpu = upload_bf16(&x);
    let result = backend
        .sigmoid_bf16_f32(&x_gpu, n)
        .expect("sigmoid 128 elements");

    assert_eq!(result.len(), n);
    check("sigmoid_bf16_f32_128", &download_f32(&result), &expected);
}
