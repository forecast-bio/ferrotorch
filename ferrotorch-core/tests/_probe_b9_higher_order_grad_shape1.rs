//! Probe — Bugfix Batch 9 / Dispatch A1 / issue #814.
//!
//! Demonstrates the bug surface end-to-end and verifies the §3 fix:
//! `reduce_grad_to_shape` must handle rank-mismatch-but-same-numel via
//! reshape (e.g. `grad []` -> `target [1]`) instead of rejecting with
//! `ShapeMismatch`. The bug shows up as a second-derivative failure on
//! shape-`[1]` leafs because the `PowBackward` chain produces a 0-D
//! intermediate gradient that gets fed into a `MulBackward` whose target
//! input shape is `[1]`.
//!
//! Pre-fix (current main):
//!   * `cited_fixture_x_cubed_at_2_second_derivative` -> ShapeMismatch
//!   * `multi_rank_x_cubed_at_2_leaf_shape_1_1_second_derivative` -> ShapeMismatch
//!
//! Post-fix:
//!   * Both reshape cases pass with d^2(x^3)/dx^2 = 6x = 12.0 at x=2.
//!   * The negative case (numel mismatch through `add` of shape `[2]`)
//!     still errors correctly.

use ferrotorch_core::autograd::higher_order::grad;
use ferrotorch_core::grad_fns::arithmetic::{add, mul, pow};
use ferrotorch_core::{Tensor, TensorStorage};

fn make_cpu_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .expect("from_storage")
}

/// Cited fixture: x = [2.0] shape [1], y = x^3, expect d^2y/dx^2 = 6x = 12.0.
///
/// This is the literal #814 surface: the second-derivative pass through
/// `PowBackward` produces an intermediate scalar (shape `[]`) that the
/// downstream `MulBackward` must reduce to target shape `[1]`. Pre-fix:
/// errors with `ShapeMismatch`. Post-fix: returns 12.0.
#[test]
fn cited_fixture_x_cubed_at_2_second_derivative() {
    let x = make_cpu_f32(&[2.0], &[1], true);
    let y = pow(&x, 3.0).expect("pow forward");

    // First derivative (create_graph=true so we can differentiate again).
    let grads = grad(&y, &[&x], true, true).expect("first grad");
    let dy_dx = grads[0].as_ref().expect("dy/dx present");
    assert_eq!(dy_dx.shape(), &[1], "first-deriv shape preserved");
    let dy_dx_data = dy_dx.data().expect("dy/dx data");
    // d(x^3)/dx = 3x^2; at x=2 -> 12.0
    assert!(
        (dy_dx_data[0] - 12.0).abs() < 1e-4,
        "first-deriv numerical: got {}",
        dy_dx_data[0]
    );

    // Second derivative — this is the surfaced bug. Pre-fix: ShapeMismatch.
    let grads2 = grad(dy_dx, &[&x], false, false).expect("second grad");
    let d2y = grads2[0].as_ref().expect("d2y present");
    assert_eq!(d2y.shape(), &[1], "second-deriv shape preserved");
    let d2y_data = d2y.data().expect("d2y data");
    // d^2(x^3)/dx^2 = 6x; at x=2 -> 12.0
    assert!(
        (d2y_data[0] - 12.0).abs() < 1e-3,
        "second-deriv numerical: got {} (expected 12.0)",
        d2y_data[0]
    );
}

/// Same shape but `[1, 1]` leaf: ensures the rank-mismatch reshape branch
/// works for grad `[]` -> target `[1, 1]` (numel 1 = 1 = 1) too. This
/// exercises the *general* `numel()` invariant, not just the specific
/// `[] -> [1]` case.
#[test]
fn multi_rank_x_cubed_leaf_shape_1_1_second_derivative() {
    let x = make_cpu_f32(&[2.0], &[1, 1], true);
    let y = pow(&x, 3.0).expect("pow forward");

    let grads = grad(&y, &[&x], true, true).expect("first grad");
    let dy_dx = grads[0].as_ref().expect("dy/dx present");
    assert_eq!(dy_dx.shape(), &[1, 1], "first-deriv shape preserved");
    assert!((dy_dx.data().expect("dy/dx data")[0] - 12.0).abs() < 1e-4);

    let grads2 = grad(dy_dx, &[&x], false, false).expect("second grad");
    let d2y = grads2[0].as_ref().expect("d2y present");
    assert_eq!(d2y.shape(), &[1, 1], "second-deriv shape preserved");
    let d2y_data = d2y.data().expect("d2y data");
    assert!(
        (d2y_data[0] - 12.0).abs() < 1e-3,
        "leaf-[1,1] second-deriv numerical: got {} (expected 12.0)",
        d2y_data[0]
    );
}

/// Same shape `[1, 1, 1]`: triple-rank reshape via `unsqueeze` chain.
#[test]
fn multi_rank_x_cubed_leaf_shape_1_1_1_second_derivative() {
    let x = make_cpu_f32(&[2.0], &[1, 1, 1], true);
    let y = pow(&x, 3.0).expect("pow forward");

    let grads = grad(&y, &[&x], true, true).expect("first grad");
    let dy_dx = grads[0].as_ref().expect("dy/dx present");
    assert_eq!(dy_dx.shape(), &[1, 1, 1]);
    assert!((dy_dx.data().expect("dy/dx data")[0] - 12.0).abs() < 1e-4);

    let grads2 = grad(dy_dx, &[&x], false, false).expect("second grad");
    let d2y = grads2[0].as_ref().expect("d2y present");
    assert_eq!(d2y.shape(), &[1, 1, 1]);
    assert!(
        (d2y.data().expect("d2y data")[0] - 12.0).abs() < 1e-3,
        "leaf-[1,1,1] second-deriv"
    );
}

/// Negative case companion (broadcast-reduction direction unaffected):
/// the existing `grad has more elements than target` code path must
/// keep working — the reshape branch must not interfere with normal
/// broadcast-reduction flow.
///
/// `add(shape=[1], shape=[2])` broadcasts to `[2]`. Backward then runs
/// `reduce_grad_to_shape(grad=[2], target=[1])` for input `a` — the
/// *good* direction (sum reduction across the broadcast axis). This
/// asserts post-fix that the reshape branch did NOT short-circuit the
/// reduction (numels differ — 2 != 1).
///
/// The direct numel-mismatch-must-error case
/// (`grad []` -> `target [2]`) is verified inline as a unit test in
/// `ferrotorch-core/src/autograd/graph.rs`
/// (`test_reduce_grad_to_shape_reshape_branch_does_not_swallow_numel_mismatch`)
/// where `reduce_grad_to_shape` is reachable as `pub(crate)`.
#[test]
fn negative_broadcast_reduction_still_works() {
    let a = make_cpu_f32(&[3.0], &[1], true);
    let b = make_cpu_f32(&[1.0, 2.0], &[2], true);
    let c = add(&a, &b).expect("[1] + [2] broadcasts");
    assert_eq!(c.shape(), &[2]);

    // Reduce to scalar via `sum` so we can grad it.
    let sum_c = mul(&c, &c).expect("c * c"); // shape [2]
    let total = ferrotorch_core::grad_fns::reduction::sum(&sum_c).expect("sum");
    assert!(total.is_scalar());

    let grads = grad(&total, &[&a, &b], false, false).expect("grad through broadcast");
    let ga = grads[0].as_ref().expect("grad a");
    let gb = grads[1].as_ref().expect("grad b");
    assert_eq!(
        ga.shape(),
        &[1],
        "broadcast-reduction direction preserved (sum across axis 0)"
    );
    assert_eq!(gb.shape(), &[2]);
}
