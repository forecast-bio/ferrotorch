//! GPU autograd integration tests.
//!
//! Verifies that `backward()` propagates gradients correctly when the
//! computation graph runs on CUDA tensors.
//!
//! Organized into two groups matching the commits that fixed them:
//!
//! 1. **GPU backward kernels** (`c627bc5`) — before this commit, backward
//!    functions like `MmBackward` called CPU-only ops (`mm`, `transpose`)
//!    on GPU tensors, hitting `GpuTensorNotAccessible`. These tests verify
//!    that backward through matmul and matmul-containing chains works on GPU.
//!
//! 2. **Shape ops graph fix** (`02caeb6`) — shape ops (squeeze, unsqueeze,
//!    flatten) used an `ensure_cpu → from_operation → restore_device` pattern
//!    that severed the computation graph on GPU, because `Tensor::to()`
//!    creates detached leaf tensors via `from_storage`. These tests verify
//!    that shape ops preserve `grad_fn` and that backward reaches leaf
//!    parameters through them.

#![cfg(feature = "cuda")]

use ferrotorch_core::{Device, Tensor, TensorStorage, backward};
use ferrotorch_gpu::init_cuda_backend;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Create a leaf tensor on GPU with requires_grad.
fn gpu_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    let t = Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true)
        .expect("cpu tensor");
    t.to(Device::Cuda(0)).expect("to GPU")
}

/// Create a constant tensor on GPU (no grad).
fn gpu_const(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    let t = Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor");
    t.to(Device::Cuda(0)).expect("to GPU")
}

/// Pull grad off a (possibly GPU) tensor and return it as a CPU f32 vec.
fn grad_data(t: &Tensor<f32>) -> Vec<f32> {
    let g = t.grad().expect("grad access").expect("grad must be Some");
    let cpu = if g.is_cuda() {
        g.cpu().expect("cpu")
    } else {
        g
    };
    cpu.data().expect("data").to_vec()
}

/// Loose float comparison.
fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "element {i}: {x} vs {y} (diff {})",
            (x - y).abs()
        );
    }
}

// ===========================================================================
// Group 1: GPU backward kernels (c627bc5)
//
// Before this commit, MmBackward::backward called the CPU-only `mm()` and
// `transpose()` functions on GPU tensors. Those functions call `.data()`
// which returns Err(GpuTensorNotAccessible) for GPU storage.
//
// The fix added GPU-native backward paths (matmul via cuBLAS, transpose
// via PTX kernel) so gradients stay on-device.
// ===========================================================================

/// Matmul backward must run entirely on GPU without hitting `.data()`.
///
/// Fails before c627bc5: MmBackward calls `mm(grad, &bt)` where both
/// tensors are on GPU. The old `mm()` calls `.data()` on a GPU tensor →
/// `Err(GpuTensorNotAccessible)`.
#[test]
fn gpu_mm_backward() {
    ensure_cuda();

    // [2,3] @ [3,2] -> [2,2]
    let a = gpu_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = gpu_leaf(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
    let c = a.mm(&b).expect("mm");

    // Forward result must stay on GPU — no silent eviction to CPU.
    assert!(c.is_cuda(), "mm output must remain on GPU");

    let loss = c.sum_all().expect("sum");
    assert!(loss.is_cuda(), "sum output must remain on GPU");

    backward(&loss).expect("backward");

    // d(sum(A@B))/dA = ones @ B^T
    // B^T = [[1,0,1],[0,1,1]], ones_col = [1,1]
    // dA row = B^T @ ones_col = [1+0, 0+1, 1+1] = [1, 1, 2]
    approx_eq(&grad_data(&a), &[1.0, 1.0, 2.0, 1.0, 1.0, 2.0], 1e-5);

    // d(sum(A@B))/dB = A^T @ ones
    // A^T = [[1,4],[2,5],[3,6]], ones_col = [1,1]
    // dB row0 = [1+4, 1+4] = [5, 5], row1 = [2+5, 2+5] = [7, 7], row2 = [3+6, 3+6] = [9, 9]
    approx_eq(&grad_data(&b), &[5.0, 5.0, 7.0, 7.0, 9.0, 9.0], 1e-5);
}

/// A chain with matmul exercises the GPU backward kernel in context.
///
/// leaf → mul(2) → mm(ones) → sum → backward
///
/// Fails before c627bc5: backward through mm hits GpuTensorNotAccessible.
#[test]
fn gpu_mm_backward_in_chain() {
    ensure_cuda();

    let x = gpu_leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let two = gpu_const(&[2.0; 4], &[2, 2]);
    let scaled = (&x * &two).expect("mul");
    assert!(scaled.is_cuda(), "mul output must remain on GPU");

    let ones = gpu_const(&[1.0, 1.0], &[2, 1]);
    let row_sums = scaled.mm(&ones).expect("mm"); // [2, 1]
    assert!(row_sums.is_cuda(), "mm output must remain on GPU");

    let loss = row_sums.sum_all().expect("sum");
    backward(&loss).expect("backward");

    // d(loss)/d(x) = 2 * ones (scaling from mul, summed by mm)
    approx_eq(&grad_data(&x), &[2.0; 4], 1e-5);
}

// ===========================================================================
// Group 2: Shape ops graph fix (02caeb6)
//
// Shape ops (squeeze, unsqueeze, flatten) used ensure_cpu/restore_device
// which called Tensor::to() to move the result back to GPU. But to()
// creates a detached leaf via from_storage — severing the computation
// graph. On CPU this was a no-op (to(Cpu) returns self.clone()), so
// the bug only manifested on GPU.
//
// The fix replaced these with view_operation(), which shares storage
// via Arc::clone and attaches a grad_fn — zero-copy and graph-preserving.
// ===========================================================================

/// Squeeze on a GPU tensor must attach a grad_fn and remain non-leaf.
///
/// Fails before 02caeb6: restore_device calls to(Cuda) which creates a
/// detached leaf. grad_fn() returns None, is_leaf() returns true.
#[test]
fn gpu_squeeze_preserves_graph() {
    ensure_cuda();

    let x = gpu_leaf(&[1.0, 2.0, 3.0], &[1, 3]);
    let squeezed = x.squeeze_t(0).expect("squeeze");

    assert_eq!(squeezed.shape(), &[3]);
    assert!(squeezed.is_cuda(), "squeeze output must stay on GPU");
    assert!(
        squeezed.grad_fn().is_some(),
        "squeeze must attach grad_fn on GPU"
    );
    assert!(
        !squeezed.is_leaf(),
        "squeeze output must be non-leaf on GPU"
    );
}

/// Unsqueeze on a GPU tensor must attach a grad_fn and remain non-leaf.
///
/// Fails before 02caeb6: same restore_device issue as squeeze.
#[test]
fn gpu_unsqueeze_preserves_graph() {
    ensure_cuda();

    let x = gpu_leaf(&[1.0, 2.0, 3.0], &[3]);
    let unsqueezed = x.unsqueeze_t(0).expect("unsqueeze");

    assert_eq!(unsqueezed.shape(), &[1, 3]);
    assert!(unsqueezed.is_cuda(), "unsqueeze output must stay on GPU");
    assert!(
        unsqueezed.grad_fn().is_some(),
        "unsqueeze must attach grad_fn on GPU"
    );
    assert!(
        !unsqueezed.is_leaf(),
        "unsqueeze output must be non-leaf on GPU"
    );
}

/// Flatten on a GPU tensor must attach a grad_fn and remain non-leaf.
///
/// Fails before 02caeb6: same restore_device issue as squeeze.
#[test]
fn gpu_flatten_preserves_graph() {
    ensure_cuda();

    let x = gpu_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let flat = x.flatten_t().expect("flatten");

    assert_eq!(flat.shape(), &[6]);
    assert!(flat.is_cuda(), "flatten output must stay on GPU");
    assert!(
        flat.grad_fn().is_some(),
        "flatten must attach grad_fn on GPU"
    );
    assert!(!flat.is_leaf(), "flatten output must be non-leaf on GPU");
}

/// Backward through squeeze must reach leaf parameters on GPU.
///
/// Fails before 02caeb6: squeeze output is a detached leaf, so backward
/// stops there and the original leaf parameter gets no gradient.
#[test]
fn gpu_squeeze_backward() {
    ensure_cuda();

    let x = gpu_leaf(&[1.0, 2.0, 3.0], &[3, 1]);
    let squeezed = x.squeeze_t(1).expect("squeeze");
    let loss = squeezed.sum_all().expect("sum");

    backward(&loss).expect("backward");

    approx_eq(&grad_data(&x), &[1.0, 1.0, 1.0], 1e-6);
}

/// Backward through unsqueeze must reach leaf parameters on GPU.
///
/// Fails before 02caeb6: same graph-severing issue.
#[test]
fn gpu_unsqueeze_backward() {
    ensure_cuda();

    let x = gpu_leaf(&[1.0, 2.0, 3.0], &[3]);
    let unsqueezed = x.unsqueeze_t(0).expect("unsqueeze");
    let loss = unsqueezed.sum_all().expect("sum");

    backward(&loss).expect("backward");

    approx_eq(&grad_data(&x), &[1.0, 1.0, 1.0], 1e-6);
}

/// Backward through flatten must reach leaf parameters on GPU.
///
/// Fails before 02caeb6: same graph-severing issue.
#[test]
fn gpu_flatten_backward() {
    ensure_cuda();

    let x = gpu_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let loss = x.flatten_t().expect("flatten").sum_all().expect("sum");

    backward(&loss).expect("backward");

    approx_eq(&grad_data(&x), &[1.0; 6], 1e-6);
}

/// The FF goodness pattern: leaf → mul → mm → squeeze → sum → backward.
///
/// Exercises BOTH fixes: mm backward needs GPU kernels (Group 1), and
/// squeeze must preserve the graph (Group 2). Fails before either commit.
#[test]
fn gpu_ff_goodness_pattern() {
    ensure_cuda();

    let x = gpu_leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let two = gpu_const(&[2.0; 6], &[3, 2]);
    let scaled = (&x * &two).expect("mul");

    let ones = gpu_const(&[1.0, 1.0], &[2, 1]);
    let row_sums = scaled.mm(&ones).expect("mm"); // [3, 1]
    let squeezed = row_sums.squeeze_t(1).expect("squeeze"); // [3]

    assert!(
        squeezed.grad_fn().is_some(),
        "squeeze must preserve graph on GPU"
    );

    let loss = squeezed.sum_all().expect("sum");
    backward(&loss).expect("backward");

    // d(loss)/d(x) = 2 (from the scaling) * 1 (from the sum)
    approx_eq(&grad_data(&x), &[2.0; 6], 1e-5);
}
