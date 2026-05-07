//! Conformance — `ferrotorch_profiler::flops` module parity against `torch.profiler`.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/826>.
//!
//! Verifies that [`ferrotorch_profiler::flops::estimate`] produces FLOP counts
//! that match the convention used by `torch.profiler`.  Each MAC is counted as
//! 2 FLOPs (one multiply + one add), consistent with PyTorch's profiler and
//! the TensorFlow Profiler.  Estimates are shape-driven (dense compute cost).
//!
//! What is pinned:
//! * Absolute FLOP counts for well-known ops against the fixture values.
//! * `None` return for unrecognized ops.
//! * `None` return when shapes are absent or insufficient.
//!
//! What is NOT pinned: ops that have approximations (softmax, norm layers) are
//! verified against fixture values, not independently rederived — the fixture
//! is the reference.
//!
//! Reference: `torch.profiler`, `pytorch/torch/utils/flop_counter.py`.

use ferrotorch_profiler::flops;

// ---------------------------------------------------------------------------
// Fixture loader (Layer 2)
// ---------------------------------------------------------------------------

fn fixtures_json() -> serde_json::Value {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Run scripts/regenerate_profiler_fixtures.py first.",
            p.display()
        )
    });
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

/// Run one flops fixture and assert the expected value.
fn run_flops_fixture(fixture_id: &str) {
    let all = fixtures_json();
    let fixtures = all["flops_fixtures"]
        .as_array()
        .expect("fixtures.json missing flops_fixtures");
    let fix = fixtures
        .iter()
        .find(|f| f["id"].as_str() == Some(fixture_id))
        .unwrap_or_else(|| panic!("fixture {fixture_id:?} not found"));

    let op = fix["op"].as_str().expect("op");
    let input_shapes: Vec<Vec<usize>> = fix["input_shapes"]
        .as_array()
        .expect("input_shapes")
        .iter()
        .map(|s| {
            s.as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect()
        })
        .collect();
    let expected: Option<u64> = fix["expected_flops"].as_u64();

    let got = flops::estimate(op, &input_shapes);
    assert_eq!(
        got, expected,
        "flops fixture {fixture_id}: op={op:?} shapes={input_shapes:?} \
         got {got:?} expected {expected:?}"
    );
}

// ---------------------------------------------------------------------------
// Tests: one per flops fixture
// ---------------------------------------------------------------------------

#[test]
fn flops_add_2d() {
    // elementwise add [3,4]+[3,4]: 12 elements × 1 FLOP = 12
    run_flops_fixture("flops_add_2d");
}

#[test]
fn flops_matmul_2d() {
    // [4,5] @ [5,6]: 2*4*6*5 = 240 FLOPs
    run_flops_fixture("flops_matmul_2d");
}

#[test]
fn flops_matmul_batched() {
    // [3,4,5] @ [3,5,6]: 3*2*4*6*5 = 720 FLOPs
    run_flops_fixture("flops_matmul_batched");
}

#[test]
fn flops_relu() {
    // relu [10,10]: 100 elements × 1 FLOP = 100
    run_flops_fixture("flops_relu");
}

#[test]
fn flops_conv2d() {
    // conv2d batch=1 C_in=3 C_out=16 kernel=3×3 spatial=32×32: 884 736 FLOPs
    run_flops_fixture("flops_conv2d");
}

#[test]
fn flops_layer_norm() {
    // layer_norm [32,768]: 8*32*768 = 196 608 FLOPs
    run_flops_fixture("flops_layer_norm");
}

#[test]
fn flops_softmax() {
    // softmax [2,5]: 5*numel = 5*10 = 50 FLOPs (torch.profiler approximation)
    run_flops_fixture("flops_softmax");
}

#[test]
fn flops_sum_reduction() {
    // sum [10,10]: numel-1 = 99 adds
    run_flops_fixture("flops_sum_reduction");
}

#[test]
fn flops_unknown_op_returns_none() {
    // torch.profiler returns 0/None for unrecognized ops; ferrotorch must match.
    run_flops_fixture("flops_unknown_op");
}

// ---------------------------------------------------------------------------
// Additional structural tests not covered by fixtures
// ---------------------------------------------------------------------------

#[test]
fn flops_no_shapes_returns_none() {
    // Both binary and unary ops must return None when shapes are absent.
    assert_eq!(
        flops::estimate("add", &[]),
        None,
        "add with no shapes must return None (torch.profiler: no estimate without shapes)"
    );
    assert_eq!(
        flops::estimate("relu", &[]),
        None,
        "relu with no shapes must return None"
    );
}

#[test]
fn flops_matmul_inner_dim_mismatch_returns_none() {
    // Mismatched inner dimensions → no valid matmul → None.
    let shapes = vec![vec![4usize, 5], vec![6usize, 7]];
    assert_eq!(
        flops::estimate("matmul", &shapes),
        None,
        "matmul with inner dimension mismatch must return None"
    );
}

#[test]
fn flops_matmul_1d_dot_product() {
    // 1-D dot product [5]·[5]: 2*1*1*5 = 10 FLOPs
    let shapes = vec![vec![5usize], vec![5usize]];
    assert_eq!(
        flops::estimate("matmul", &shapes),
        Some(10),
        "1-D dot product [5]·[5] must be 10 FLOPs"
    );
}

#[test]
fn flops_elementwise_binary_broadcast() {
    // add [3,4]+[1,4]: output is [3,4] = 12 elements (broadcast max numel)
    let shapes = vec![vec![3usize, 4], vec![1usize, 4]];
    assert_eq!(
        flops::estimate("add", &shapes),
        Some(12),
        "broadcast add: output numel is max(12,4) = 12 FLOPs"
    );
}

#[test]
fn flops_sub_mul_div_same_as_add() {
    // All elementwise binary ops count identically (1 FLOP per output element).
    let shapes = vec![vec![4usize, 4], vec![4usize, 4]];
    let expected = Some(16u64);
    for op in &["sub", "mul", "div"] {
        assert_eq!(
            flops::estimate(op, &shapes),
            expected,
            "{op} with [4,4]×[4,4] must be 16 FLOPs"
        );
    }
}

#[test]
fn flops_activation_variants_all_recognized() {
    // sigmoid, tanh, gelu, silu, leaky_relu must all produce estimates
    // (not None) when a valid shape is given. Exact counts may differ from
    // torch.profiler's per-op approximation, but must be non-None.
    let shapes = vec![vec![8usize, 8]];
    for op in &["sigmoid", "tanh", "gelu", "silu", "leaky_relu"] {
        assert!(
            flops::estimate(op, &shapes).is_some(),
            "{op} must return Some(flops) for a valid shape"
        );
    }
}

#[test]
fn flops_conv1d() {
    // conv1d batch=1 C_in=4 C_out=8 kernel=3 spatial=100:
    // 2*1*8*4*3*100 = 19 200 FLOPs
    let shapes = vec![vec![1usize, 4, 100], vec![8usize, 4, 3]];
    assert_eq!(
        flops::estimate("conv1d", &shapes),
        Some(19_200),
        "conv1d [1,4,100]×[8,4,3] must be 19 200 FLOPs"
    );
}

#[test]
fn flops_mm_alias_same_as_matmul() {
    // "mm" must be recognized as a matmul variant.
    let shapes = vec![vec![4usize, 5], vec![5usize, 6]];
    assert_eq!(
        flops::estimate("mm", &shapes),
        Some(240),
        "mm alias must produce the same estimate as matmul for [4,5]@[5,6]"
    );
}

#[test]
fn flops_mean_reduction() {
    // mean [10,10]: numel-1 = 99 (same formula as sum — reduction count)
    let shapes = vec![vec![10usize, 10]];
    assert_eq!(
        flops::estimate("mean", &shapes),
        Some(99),
        "mean [10,10]: numel-1 = 99"
    );
}

#[test]
fn flops_pow_double_per_element() {
    // pow uses 2 FLOPs per element.
    let shapes = vec![vec![5usize, 5]];
    assert_eq!(
        flops::estimate("pow", &shapes),
        Some(50),
        "pow [5,5]: 2 * 25 = 50 FLOPs"
    );
}
